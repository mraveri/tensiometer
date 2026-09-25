"""
Minimal training loop for the normalizing flows, with Keras-like callbacks.

:class:`Trainer` optimizes the parameters of a module with Adam by minimizing a loss of the
log probabilities of batches of samples, evaluates the validation loss after every epoch,
stops on non-finite losses, and calls :class:`Callback` hooks (used by the flows for
monitoring and by the learning rate schedulers).
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import torch
import tqdm

from . import tensor_utilities as tu
from .trainable_bijectors import reset_module_parameters

__all__ = ['Callback', 'History', 'Trainer', 'reset_module_parameters', 'count_parameters']

###############################################################################
# callbacks:


class Callback(object):
    """
    Base class of the training callbacks. The trainer sets ``self.trainer`` before training
    and calls the hooks below; all of them do nothing by default.
    """

    trainer = None

    def set_trainer(self, trainer):
        """
        Attach the callback to a trainer.

        :param trainer: :class:`Trainer` running the fit.
        :returns: None.
        """
        self.trainer = trainer

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        :param logs: dictionary of logs, empty when called by :class:`Trainer`.
        :returns: None.
        """

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        :param logs: dictionary of logs, empty when called by :class:`Trainer`.
        :returns: None.
        """

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of every epoch.

        :param epoch: zero-based index of the epoch.
        :param logs: dictionary of logs, empty when called by :class:`Trainer`.
        :returns: None.
        """

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of every epoch with the epoch logs (``loss``, ``val_loss``, ``lr``).

        :param epoch: zero-based index of the epoch.
        :param logs: dictionary with the mean training batch loss ``loss``, the validation
            loss ``val_loss`` (only with validation data) and the learning rate ``lr``.
        :returns: None.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """
        Called before every optimization step.

        :param batch: zero-based index of the step within the epoch.
        :param logs: dictionary of logs, empty when called by :class:`Trainer`.
        :returns: None.
        """

    def on_train_batch_end(self, batch, logs=None):
        """
        Called after every optimization step with the batch loss.

        :param batch: zero-based index of the step within the epoch.
        :param logs: dictionary with the batch loss ``loss``.
        :returns: None.
        """


class History(Callback):
    """
    Record the epoch logs. ``history`` maps every log key to the list of its values.
    """

    def __init__(self):
        self.history = {}
        self.epoch = []

    def on_train_begin(self, logs=None):
        """
        Reset the list of epochs.

        :param logs: dictionary of logs, unused.
        :returns: None.
        """
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Append the epoch logs.

        :param epoch: zero-based index of the epoch, appended to ``epoch``.
        :param logs: dictionary of epoch logs; each value is appended to ``history[key]``.
        :returns: None.
        """
        logs = logs or {}
        self.epoch.append(epoch)
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)

###############################################################################
# helpers:


def count_parameters(module):
    """
    Number of trainable scalar parameters of a module.

    :param module: ``torch.nn.Module``.
    :returns: integer count.
    """
    return int(sum(_p.numel() for _p in module.parameters() if _p.requires_grad))


def _as_optional_tensor(value, device):
    """Convert to a tensor on ``device``, keeping None."""
    if value is None:
        return None
    return tu.to_tensor(value, device=device)

###############################################################################
# trainer:


class Trainer(object):
    """
    Train a module by minimizing ``loss(y, log_prob_fn(x), sample_weight=w)``.

    :param module: ``torch.nn.Module`` holding the trainable parameters.
    :param log_prob_fn: callable mapping a batch of samples to model outputs.
    :param loss: callable ``loss(y_true, y_pred, sample_weight=None)`` returning a scalar.
    :param learning_rate: initial learning rate of Adam.
    :param global_clipnorm: optional maximum global norm of the gradients.
    """

    def __init__(self, module, log_prob_fn, loss, learning_rate=1e-3, global_clipnorm=None):
        self.module = module
        self.log_prob_fn = log_prob_fn
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.global_clipnorm = global_clipnorm
        self.stop_training = False
        self.optimizer = None
        self.reset_optimizer()

    def _trainable_parameters(self):
        """List of the trainable parameters."""
        return [_p for _p in self.module.parameters() if _p.requires_grad]

    def reset_optimizer(self):
        """
        Create a fresh Adam optimizer (Keras epsilon, 1e-7). The optimizer is None when the
        module has no trainable parameters.
        """
        parameters = self._trainable_parameters()
        if len(parameters) == 0:
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate, eps=1e-7)

    @property
    def device(self):
        """Device of the module (the default device if it holds no tensors)."""
        return tu.module_device(self.module, default=tu.get_device())

    def get_learning_rate(self):
        """
        Current learning rate.

        :returns: float, or None without optimizer.
        """
        if self.optimizer is None:
            return None
        return float(self.optimizer.param_groups[0]['lr'])

    def _batch_loss(self, x, y, w):
        """Loss of one batch."""
        y_pred = self.log_prob_fn(x)
        return self.loss(y, y_pred, sample_weight=w)

    def _index_stream(self, n, num_indices, device):
        """Shuffled indices, concatenating permutations when more than ``n`` are needed."""
        blocks = []
        total = 0
        while total < num_indices:
            blocks.append(torch.randperm(n, device=device))
            total += n
        return torch.cat(blocks)[:num_indices]

    def evaluate(self, x, y=None, sample_weight=None, batch_size=None):
        """
        Loss on a data set, averaged over batches of ``batch_size`` samples weighted by their size.

        :param x: samples.
        :param y: optional targets.
        :param sample_weight: optional sample weights.
        :param batch_size: evaluation batch size, defaults to the whole set.
        :returns: float loss.
        """
        device = self.device
        x = tu.to_tensor(x, device=device)
        y = _as_optional_tensor(y, device)
        sample_weight = _as_optional_tensor(sample_weight, device)
        n = x.shape[0]
        if n == 0:
            return float('nan')
        if batch_size is None or batch_size <= 0:
            batch_size = n
        total, count = 0., 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                _y = None if y is None else y[start:stop]
                _w = None if sample_weight is None else sample_weight[start:stop]
                value = float(self._batch_loss(x[start:stop], _y, _w))
                total += value * (stop - start)
                count += stop - start
        return total / count

    def fit(self, x, y=None, sample_weight=None, validation_data=None, epochs=1, batch_size=None,
            steps_per_epoch=None, callbacks=None, verbose=1):
        """
        Run the training loop.

        :param x: training samples ``(N, D)``.
        :param y: optional training targets passed to the loss as ``y_true``.
        :param sample_weight: optional training weights ``(N,)``.
        :param validation_data: optional tuple ``(x, y)`` or ``(x, y, sample_weight)``.
        :param epochs: number of epochs.
        :param batch_size: batch size, defaults to 32.
        :param steps_per_epoch: optimization steps per epoch, defaults to ``ceil(N / batch_size)``.
        :param callbacks: list of :class:`Callback`.
        :param verbose: 0 silent, 1 or 2 one line per epoch, -1 progress bar.
        :returns: :class:`History` with the epoch logs.
        :raises ValueError: if the module has no trainable parameters.

        A non-finite batch loss stops training without taking that optimization step.
        The interrupted epoch is logged from its finite batch losses only, and is not
        logged at all if it had none (the parameters are then unchanged since the
        previous logged epoch).
        """
        if self.optimizer is None:
            raise ValueError('Cannot train a module without trainable parameters.')
        device = self.device
        x = tu.to_tensor(x, device=device)
        y = _as_optional_tensor(y, device)
        sample_weight = _as_optional_tensor(sample_weight, device)
        n = x.shape[0]
        if batch_size is None:
            batch_size = 32
        batch_size = max(int(batch_size), 1)
        if steps_per_epoch is None:
            steps_per_epoch = int(np.ceil(n / batch_size))
        steps_per_epoch = max(int(steps_per_epoch), 1)
        # validation data:
        validation = None
        if validation_data is not None:
            validation = list(validation_data) + [None] * (3 - len(validation_data))
        # callbacks:
        history = History()
        callbacks = list(callbacks or []) + [history]
        for callback in callbacks:
            callback.set_trainer(self)
        self.stop_training = False
        # progress bar:
        progress = None
        if verbose == -1:
            progress = tqdm.tqdm(total=epochs, desc='Training', leave=True)
        # training loop:
        for callback in callbacks:
            callback.on_train_begin({})
        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch, {})
            indices = self._index_stream(n, steps_per_epoch * batch_size, device)
            batch_losses = []
            for step in range(steps_per_epoch):
                for callback in callbacks:
                    callback.on_train_batch_begin(step, {})
                batch = indices[step * batch_size:(step + 1) * batch_size]
                _x = x[batch]
                _y = None if y is None else y[batch]
                _w = None if sample_weight is None else sample_weight[batch]
                self.optimizer.zero_grad(set_to_none=True)
                value = self._batch_loss(_x, _y, _w)
                loss_value = float(value.detach())
                if not np.isfinite(loss_value):
                    print('Batch {}: Invalid loss, terminating training'.format(step))
                    self.stop_training = True
                    break
                batch_losses.append(loss_value)
                value.backward()
                if self.global_clipnorm is not None:
                    torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), float(self.global_clipnorm))
                self.optimizer.step()
                for callback in callbacks:
                    callback.on_train_batch_end(step, {'loss': loss_value})
            # epoch logs, skipped if the epoch was aborted before any finite step:
            if len(batch_losses) > 0:
                self._end_epoch(epoch, epochs, batch_losses, validation, batch_size, callbacks, progress, verbose)
            if self.stop_training:
                break
        for callback in callbacks:
            callback.on_train_end({})
        if progress is not None:
            progress.close()
        return history

    def _end_epoch(self, epoch, epochs, batch_losses, validation, batch_size, callbacks, progress, verbose):
        """Build the epoch logs from the finite batch losses, call the epoch-end hooks and print feedback."""
        logs = {'loss': float(np.mean(batch_losses))}
        if validation is not None:
            logs['val_loss'] = self.evaluate(validation[0], validation[1], validation[2], batch_size=batch_size)
        logs['lr'] = self.get_learning_rate()
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)
        # feedback:
        if progress is not None:
            progress.update(1)
            progress.set_postfix({_k: _v for _k, _v in logs.items() if _k in ['loss', 'val_loss']})
        elif verbose in (1, 2):
            _message = 'Epoch {}/{} - loss: {:.4f}'.format(epoch + 1, epochs, logs['loss'])
            if 'val_loss' in logs:
                _message += ' - val_loss: {:.4f}'.format(logs['val_loss'])
            _message += ' - lr: {:.3g}'.format(logs['lr'])
            print(_message)
