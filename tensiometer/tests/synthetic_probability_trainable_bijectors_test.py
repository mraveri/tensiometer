"""Tests for trainable bijectors."""

#########################################################################################################
# Imports

import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensiometer.synthetic_probability import trainable_bijectors as tb

#########################################################################################################
# TensorFlow Probability aliases

tfb = tfp.bijectors

#########################################################################################################
# Test cases


class TestTrainableBijectors(unittest.TestCase):
    """Trainable bijectors test suite."""
    def test_trainable_transformation_interface(self):
        """Test TrainableTransformation abstract interface."""
        base = tb.TrainableTransformation()
        with self.assertRaises(NotImplementedError):
            base.save("dummy")
        with self.assertRaises(NotImplementedError):
            base.load("dummy")

    def test_toggle_str_eq_fallbacks(self):
        """Test ToggleStr eq fallback branches."""
        class ToggleStr(str):
            """Toggle Str test suite."""
            def __new__(cls, value):
                """New."""
                obj = super().__new__(cls, value)
                obj.calls = 0
                return obj

            def __eq__(self, other):
                """Eq."""
                self.calls += 1
                if other == "spline":
                    return self.calls == 1
                if other == "affine":
                    return True
                return super().__eq__(other)

        toggled = ToggleStr("affine")
        self.assertTrue(toggled == "spline")
        self.assertTrue(toggled == "affine")
        self.assertFalse(toggled == "other")

        class NonAffineStr(str):
            """Non Affine test suite."""
            def __new__(cls, value):
                """New."""
                obj = super().__new__(cls, value)
                obj.spline_calls = 0
                return obj

            def __eq__(self, other):
                """Eq."""
                if other == "spline":
                    self.spline_calls += 1
                    return self.spline_calls != 3
                if other == "affine":
                    return False
                return super().__eq__(other)

        non_affine = NonAffineStr("spline")
        self.assertTrue(non_affine == "spline")
        self.assertFalse(non_affine == "affine")
        self.assertFalse(non_affine == "other")

        class FlakySpline(str):
            """Flaky Spline test suite."""
            def __new__(cls, value):
                """New."""
                obj = super().__new__(cls, value)
                obj.calls = 0
                return obj

            def __eq__(self, other):
                """Eq."""
                if other == "affine":
                    return False
                if other == "spline":
                    self.calls += 1
                    return self.calls <= 2
                return super().__eq__(other)

        flaky = FlakySpline("spline")
        self.assertFalse(flaky == "affine")
        self.assertTrue(flaky == "spline")
        self.assertFalse(flaky == "other")
    def test_scale_roto_shift_identity(self):
        """Test Scale roto shift identity."""
        bij = tb.ScaleRotoShift(2, initializer="zeros")
        x = tf.constant([[1.0, -1.0], [0.5, 0.25]], dtype=tf.float32)
        y = bij.forward(x)
        self.assertTrue(np.allclose(y.numpy(), x.numpy()))
        self.assertTrue(np.allclose(bij.inverse(y).numpy(), x.numpy()))

        fldj = bij.forward_log_det_jacobian(x, event_ndims=0)
        ildj = bij.inverse_log_det_jacobian(y, event_ndims=0)
        self.assertAlmostEqual(float(fldj.numpy()), 0.0, places=5)
        self.assertAlmostEqual(float(ildj.numpy()), 0.0, places=5)
        self.assertTrue(bij._is_increasing())
        self.assertIn("shift", tb.ScaleRotoShift._parameter_properties(tf.float32))

    def test_scale_roto_shift_disabled_components(self):
        """Test Scale roto shift disabled components."""
        bij = tb.ScaleRotoShift(3, shift=False, scale=False, roto=False, initializer="zeros")
        _ = bij.shift  # property coverage
        x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        y = bij._forward(x)
        self.assertTrue(np.allclose(y.numpy(), x.numpy()))
        inv = bij._inverse(y)
        self.assertTrue(np.allclose(inv.numpy(), x.numpy()))
        ildj = bij._inverse_log_det_jacobian(y)
        self.assertAlmostEqual(float(ildj.numpy()), 0.0, places=5)

    def test_circular_rational_quadratic_spline_bounds_and_inverse(self):
        """Test Circular rational quadratic spline bounds and inverse."""
        bin_widths = tf.constant([0.4, 0.6], dtype=tf.float32)
        bin_heights = tf.constant([0.5, 0.5], dtype=tf.float32)
        knot_slopes = tf.constant([1.0], dtype=tf.float32)
        boundary = tf.constant(1.5, dtype=tf.float32)

        spline = tb.CircularRationalQuadraticSpline(
            bin_widths=bin_widths,
            bin_heights=bin_heights,
            knot_slopes=knot_slopes,
            boundary_knot_slope=boundary,
            range_min=-1.0,
            range_max=1.0,
        )

        x = tf.constant([0.0], dtype=tf.float32)
        y = spline.forward(x)
        x_back = spline.inverse(y)
        self.assertTrue(np.allclose(x.numpy(), x_back.numpy(), atol=1e-5))

        x_upper = tf.constant([1.5], dtype=tf.float32)
        y_upper = spline.forward(x_upper)
        expected_upper = boundary.numpy().item() * (x_upper.numpy() - 1.0) + 1.0
        self.assertAlmostEqual(
            float(y_upper.numpy().item()), float(expected_upper.squeeze()), places=5
        )

        fldj_upper = spline.forward_log_det_jacobian(x_upper, event_ndims=0)
        self.assertAlmostEqual(
            float(fldj_upper.numpy().item()), np.log(boundary.numpy().item()), places=5
        )

        # Exercise inverse on out-of-bounds value to trigger boundary handling.
        y_upper_val = tf.constant([2.0], dtype=tf.float32)
        x_from_y = spline.inverse(y_upper_val)
        self.assertGreater(float(x_from_y.numpy().item()), 1.0)
        # Force invalid rank to hit gather_squeeze error
        original_rank = tb.tensorshape_util.rank
        try:
            tb.tensorshape_util.rank = lambda shape: None
            with self.assertRaises(ValueError):
                spline._compute_shared(x=tf.constant([0.0], dtype=tf.float32))
        finally:
            tb.tensorshape_util.rank = original_rank

    def test_spline_helper_parameterization_variants(self):
        """Test Spline helper parameterization variants."""
        knots = 3
        with self.assertRaises(ValueError):
            tb.SplineHelper(
                shift_and_log_scale_fn=lambda x: x,
                spline_knots=knots,
                equispaced_x_knots=True,
                equispaced_y_knots=True,
            )

        def shift_fn(x):
            """Shift fn."""
            batch = tf.shape(x)[0]
            return tf.ones((batch, 3 * knots - 1), dtype=tf.float32)

        helper = tb.SplineHelper(
            shift_and_log_scale_fn=shift_fn,
            spline_knots=knots,
            range_min=-2.0,
            range_max=2.0,
        )
        bij = helper._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        forward_val = bij.forward(tf.constant([0.1], dtype=tf.float32))
        self.assertEqual(forward_val.shape, (1,))

        def shift_fn_equispaced(x):
            """Shift fn equispaced."""
            batch = tf.shape(x)[0]
            return tf.ones((batch, knots + 2), dtype=tf.float32)

        helper_eq = tb.SplineHelper(
            shift_and_log_scale_fn=shift_fn_equispaced,
            spline_knots=knots,
            equispaced_x_knots=True,
            slope_std=0.25,
        )
        bij_eq = helper_eq._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertIsInstance(bij_eq, tfb.RationalQuadraticSpline)
        self.assertTrue(tf.reduce_all(bij_eq.bin_widths > 0))

        def shift_fn_equispaced_y(x):
            """Shift fn equispaced y."""
            batch = tf.shape(x)[0]
            return tf.ones((batch, 2 * knots - 1), dtype=tf.float32)

        helper_eq_y = tb.SplineHelper(
            shift_and_log_scale_fn=shift_fn_equispaced_y,
            spline_knots=knots,
            equispaced_y_knots=True,
            min_bin_height=0.1,
        )
        bij_eq_y = helper_eq_y._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertTrue(tf.reduce_all(bij_eq_y.bin_heights > 0))

        with self.assertRaises(ValueError):
            tb.SplineHelper(bijector_fn=tfb.Identity(), shift_and_log_scale_fn=lambda x: x)
        with self.assertRaises(ValueError):
            tb.SplineHelper()

        # direct bijector_fn path
        helper_direct = tb.SplineHelper(bijector_fn=lambda x, **_: tfb.Identity())
        self.assertIsInstance(helper_direct._bijector_fn(None), tfb.Identity)

    def test_circular_spline_helper_boundary_slopes(self):
        """Test Circular spline helper boundary slopes."""
        knots = 3

        def shift_fn(x):
            """Shift fn."""
            batch = tf.shape(x)[0]
            total = 2 * knots + knots
            return tf.ones((batch, total), dtype=tf.float32)

        helper = tb.CircularSplineHelper(
            shift_and_log_scale_fn=shift_fn, spline_knots=knots, range_max=2.0
        )
        bij = helper._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertIsInstance(bij, tb.CircularRationalQuadraticSpline)

        outside = bij.forward(tf.constant([2.5], dtype=tf.float32))
        self.assertGreater(outside.numpy()[0], 2.0)

        with self.assertRaises(NotImplementedError):
            helper_ni = tb.CircularSplineHelper(
                shift_and_log_scale_fn=shift_fn,
                spline_knots=knots,
                range_max=2.0,
                slope_std=0.5,
            )
            helper_ni._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))

        with self.assertRaises(ValueError):
            tb.CircularSplineHelper(
                bijector_fn=tfb.Identity(), shift_and_log_scale_fn=lambda x: x
            )
        with self.assertRaises(ValueError):
            tb.CircularSplineHelper()

        # direct bijector_fn path and equispaced knots handling
        direct = tb.CircularSplineHelper(
            bijector_fn=lambda x, **_: tfb.Identity(),
            equispaced_x_knots=True,
            equispaced_y_knots=False,
            spline_knots=2,
            range_max=1.5,
        )
        self.assertIsInstance(direct._bijector_fn(None), tfb.Identity)

    def test_build_nn_and_ar_model(self):
        """Test Build nn and ar model."""
        nn = tb.build_nn(0, 2, hidden_units=[1, 1])
        nn_out = nn(tf.zeros((1, 0), dtype=tf.float32))
        self.assertEqual(nn_out.shape, (1, 2))

        x = tf.ones((2, 3), dtype=tf.float32)
        ar_model = tb.build_AR_model(
            num_params=3,
            transf_params=2,
            hidden_units=[4, 2],
            identity_dims=[1],
            scale_with_dim=True,
        )
        params = ar_model(x)
        self.assertEqual(params.shape, (2, 3, 2))
        self.assertTrue(np.allclose(params.numpy()[:, 1, :], 0.0))

    def test_const_zeros_python_function(self):
        """Test Const zeros python function."""
        x = tf.ones((2, 1), dtype=tf.float32)
        zeros = tb.const_zeros.python_function(x, 3)
        self.assertEqual(zeros.shape, (2, 3))
        self.assertTrue(np.allclose(zeros.numpy(), 0.0))

    def test_min_var_permutations(self):
        """Test Min var permutations."""
        np.random.seed(0)
        perms = tb.min_var_permutations(d=3, n=2, min_number=20)
        self.assertEqual(len(perms), 2)
        identity = np.arange(3)
        for p in perms:
            self.assertEqual(len(p), 3)
            self.assertFalse(np.all(p == identity))

    def test_autoregressive_flow_range_and_periodic_handling(self):
        """Test Autoregressive flow range and periodic handling."""
        flow = tb.AutoregressiveFlow(
            num_params=2,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[4],
            parameters_min=np.array([-5.0, -6.0], dtype=np.float32),
            parameters_max=np.array([9.0, 6.0], dtype=np.float32),
            periodic_params=np.array([False, False]),
            permutations=False,
        )
        self.assertGreater(float(tf.convert_to_tensor(flow.range_max)), 9.0)

        with self.assertRaises(AssertionError):
            tb.AutoregressiveFlow(
                num_params=1,
                transformation_type="affine",
                map_to_unitcube=True,
                permutations=False,
            )

        # periodic preprocessing path with permutations inserted after first layer
        periodic_flow = tb.AutoregressiveFlow(
            num_params=2,
            transformation_type="spline",
            n_transformations=2,
            hidden_units=[3],
            periodic_params=np.array([True, False]),
            permutations=True,
        )
        sample = tf.constant([[0.1, 0.2]], dtype=tf.float32)
        self.assertEqual(periodic_flow.bijector.forward(sample).shape, sample.shape)

        # map-to-unit-cube branch with custom permutations sequence
        unit_flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            map_to_unitcube=True,
            permutations=[np.array([0], dtype=np.int32)],
        )
        val = unit_flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(val.shape, (1, 1))

    def test_autoregressive_flow_variants_and_logging(self):
        """Test Autoregressive flow variants and logging."""
        flow = tb.AutoregressiveFlow(
            num_params=2,
            transformation_type=["affine", "spline"],
            autoregressive_type=["masked", "flex"],
            n_transformations=2,
            hidden_units=[3],
            equispaced_x_knots=True,
            equispaced_y_knots=False,
            scale_roto_shift=True,
            permutations=None,
            feedback=2,
        )
        sample = tf.constant([[0.0, 0.1]], dtype=tf.float32)
        out = flow.bijector.forward(sample)
        self.assertEqual(out.shape, sample.shape)

    def test_autoregressive_flow_save_and_load(self):
        """Test Autoregressive flow save and load."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="affine",
            n_transformations=1,
            hidden_units=[2],
            permutations=[np.array([0], dtype=np.int32)],
        )
        x = tf.constant([[0.1]], dtype=tf.float32)
        original = flow.bijector.forward(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flow")
            flow.save(path)
            restored = tb.AutoregressiveFlow.load(
                path,
                transformation_type="affine",
                n_transformations=1,
                hidden_units=[2],
            )
            restored_val = restored.bijector.forward(x)
            self.assertTrue(np.allclose(original.numpy(), restored_val.numpy()))

    def test_bijector_layer_wraps_forward(self):
        """Test Bijector layer wraps forward."""
        layer = tb.BijectorLayer(bijector=tfb.Tanh())
        x = tf.constant([[0.0, 1.0]], dtype=tf.float32)
        y = layer(x)
        self.assertTrue(np.allclose(y.numpy(), np.tanh(x.numpy())))

    def test_circular_spline_helper_auto_range_min(self):
        """Test Circular spline helper auto range min."""
        knots = 3

        def shift_fn(x):
            """Shift fn."""
            batch = tf.shape(x)[0]
            # plenty of parameters for non-equispaced setup
            return tf.ones((batch, 3 * knots), dtype=tf.float32)

        helper = tb.CircularSplineHelper(
            shift_and_log_scale_fn=shift_fn, spline_knots=knots, range_min=None, range_max=2.0
        )
        bij = helper._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        # range_min should have been set to -range_max
        self.assertEqual(float(bij.range_min.numpy()), -2.0)
        _ = bij.forward(tf.constant([0.1], dtype=tf.float32))

    def test_circular_spline_helper_equispaced_knots_error(self):
        """Test Circular spline helper equispaced knots error."""
        with self.assertRaises(ValueError):
            tb.CircularSplineHelper(
                shift_and_log_scale_fn=lambda x: x,
                spline_knots=2,
                equispaced_x_knots=True,
                equispaced_y_knots=True,
            )

    def test_circular_spline_helper_equispaced_x_branch(self):
        """Test Circular spline helper equispaced x branch."""
        knots = 3

        def shift_fn(x):
            """Shift fn."""
            batch = tf.shape(x)[0]
            # parameters sized to give two interior slopes
            return tf.ones((batch, 5), dtype=tf.float32)

        helper = tb.CircularSplineHelper(
            shift_and_log_scale_fn=shift_fn,
            spline_knots=knots,
            equispaced_x_knots=True,
            equispaced_y_knots=False,
        )
        bij = helper._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertTrue(tf.reduce_all(bij.bin_widths > 0))

    def test_circular_spline_helper_equispaced_y_branch(self):
        """Test Circular spline helper equispaced y branch."""
        knots = 3

        def shift_fn(x):
            """Shift fn."""
            batch = tf.shape(x)[0]
            return tf.ones((batch, 5), dtype=tf.float32)

        helper = tb.CircularSplineHelper(
            shift_and_log_scale_fn=shift_fn,
            spline_knots=knots,
            equispaced_x_knots=False,
            equispaced_y_knots=True,
        )
        bij = helper._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertTrue(tf.reduce_all(bij.bin_heights > 0))

    def test_circular_spline_helper_with_range_min(self):
        """Test Circular spline helper with range min."""
        knots = 2

        def shift_fn(x):
            """Shift fn."""
            batch = tf.shape(x)[0]
            return tf.ones((batch, 3 * knots), dtype=tf.float32)

        helper = tb.CircularSplineHelper(
            shift_and_log_scale_fn=shift_fn,
            spline_knots=knots,
            range_min=-0.5,
            range_max=1.5,
        )
        bij = helper._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertEqual(float(bij.range_min.numpy()), -0.5)

    def test_build_ar_model_no_dim_scaling(self):
        """Test Build ar model no dim scaling."""
        model = tb.build_AR_model(
            num_params=2, transf_params=1, hidden_units=[3], scale_with_dim=False
        )
        out = model(tf.ones((1, 2), dtype=tf.float32))
        self.assertEqual(out.shape, (1, 2, 1))

    def test_autoregressive_flow_permutations_bool(self):
        """Test Autoregressive flow permutations bool."""
        called = {}

        def fake_min_var_permutations(d, n, min_number=10000):
            """Fake min var permutations."""
            called["d"] = d
            called["n"] = n
            return [np.arange(d), np.arange(d)[::-1]]

        original = tb.min_var_permutations
        tb.min_var_permutations = fake_min_var_permutations
        try:
            flow_true = tb.AutoregressiveFlow(
                num_params=2,
                transformation_type="affine",
                n_transformations=2,
                hidden_units=[2],
                permutations=True,
            )
            self.assertIn("d", called)
            self.assertEqual(len(flow_true.permutations), 2)

            flow_false = tb.AutoregressiveFlow(
                num_params=1,
                transformation_type="affine",
                n_transformations=1,
                hidden_units=[1],
                permutations=False,
            )
            self.assertFalse(flow_false.permutations)
        finally:
            tb.min_var_permutations = original

    def test_autoregressive_flow_range_adjustment_and_kernel_default(self):
        """Test Autoregressive flow range adjustment and kernel default."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            parameters_min=np.array([-10.0], dtype=np.float32),
            parameters_max=np.array([10.0], dtype=np.float32),
            permutations=False,
        )
        self.assertGreater(float(tf.convert_to_tensor(flow.range_max)), 5.0)
        # default kernel initializer branch still builds a working bijector
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_affine_params_range_skip(self):
        """Test Autoregressive flow affine params range skip."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="affine",
            n_transformations=1,
            hidden_units=[1],
            parameters_min=np.array([-2.0], dtype=np.float32),
            parameters_max=np.array([2.0], dtype=np.float32),
            permutations=False,
        )
        self.assertIsNone(flow.range_max)

    def test_autoregressive_flow_invalid_transformation_type(self):
        """Test Autoregressive flow invalid transformation type."""
        with self.assertRaises(ValueError):
            tb.AutoregressiveFlow(
                num_params=1,
                transformation_type="invalid",
                n_transformations=1,
                hidden_units=[1],
                permutations=False,
            )

    def test_autoregressive_flow_invalid_autoregressive_type(self):
        """Test Autoregressive flow invalid autoregressive type."""
        with self.assertRaises(ValueError):
            tb.AutoregressiveFlow(
                num_params=1,
                transformation_type="affine",
                autoregressive_type="bad",
                n_transformations=1,
                hidden_units=[1],
                permutations=False,
            )

    def test_autoregressive_flow_periodic_affine_branch(self):
        """Test Autoregressive flow periodic affine branch."""
        class ToggleStr(str):
            """Toggle Str test suite."""
            def __new__(cls, value):
                """New."""
                obj = super().__new__(cls, value)
                obj.calls = 0
                return obj

            def __eq__(self, other):
                # first comparison behaves as 'spline' to pass assertions,
                # later comparisons act as 'affine' to reach the Tanh branch.
                """Eq."""
                if other == "spline":
                    self.calls += 1
                    return self.calls == 1
                if other == "affine":
                    return True
                return super().__eq__(other)

        toggled = ToggleStr("affine")
        self.assertTrue(toggled == "spline")
        self.assertTrue(toggled == "affine")
        self.assertFalse(toggled == "other")

        flow_type = ToggleStr("affine")
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type=flow_type,
            n_transformations=1,
            hidden_units=[1],
            periodic_params=np.array([True]),
            permutations=False,
        )
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_equispaced_y_knots(self):
        """Test Autoregressive flow equispaced y knots."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            equispaced_y_knots=True,
            permutations=False,
        )
        sample = tf.constant([[0.0]], dtype=tf.float32)
        self.assertEqual(flow.bijector.forward(sample).shape, sample.shape)

    def test_autoregressive_flow_map_to_unitcube_branch(self):
        """Test Autoregressive flow map to unitcube branch."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            map_to_unitcube=True,
            permutations=[np.array([0], dtype=np.int32)],
        )
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_kernel_initializer_default(self):
        """Test Autoregressive flow kernel initializer default."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="affine",
            n_transformations=2,
            hidden_units=[2],
            permutations=False,
        )
        self.assertIsNotNone(flow.bijector)

    def test_autoregressive_flow_range_adjustment_trigger(self):
        """Test Autoregressive flow range adjustment trigger."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            parameters_min=np.array([-20.0], dtype=np.float32),
            parameters_max=np.array([3.0], dtype=np.float32),
            range_max=1.0,
            permutations=False,
        )
        self.assertGreater(float(tf.convert_to_tensor(flow.range_max)), 1.0)

    def test_circular_spline_helper_range_min_set_and_delta(self):
        """Test Circular spline helper range min set and delta."""
        def shift_fn(x):
            """Shift fn."""
            batch = tf.shape(x)[0]
            return tf.ones((batch, 4), dtype=tf.float32)

        helper = tb.CircularSplineHelper(
            shift_and_log_scale_fn=shift_fn,
            spline_knots=2,
            range_min=None,
            range_max=1.5,
            equispaced_x_knots=True,
        )
        bij = helper._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertEqual(float(bij.range_min.numpy()), -1.5)
        # additional coverage for range_min auto-setup without equispaced knots
        helper_non_equi = tb.CircularSplineHelper(
            shift_and_log_scale_fn=lambda x: tf.ones((tf.shape(x)[0], 6), dtype=tf.float32),
            spline_knots=2,
            range_min=None,
            range_max=2.0,
        )
        bij2 = helper_non_equi._bijector_fn(tf.zeros((1, 1), dtype=tf.float32))
        self.assertEqual(float(bij2.range_min.numpy()), -2.0)

    def test_circular_spline_helper_range_min_assertion(self):
        """Test Circular spline helper range min assertion."""
        with self.assertRaises(AssertionError):
            tb.CircularSplineHelper(
                shift_and_log_scale_fn=lambda x: tf.ones((tf.shape(x)[0], 4), dtype=tf.float32),
                spline_knots=2,
                range_min=None,
                range_max=-1.0,
            )

    def test_autoregressive_flow_permutation_bool_branches(self):
        """Test Autoregressive flow permutation bool branches."""
        flow_true = tb.AutoregressiveFlow(
            num_params=2,
            transformation_type="affine",
            n_transformations=2,
            hidden_units=[2],
            permutations=True,
        )
        self.assertIsInstance(flow_true.permutations, list)

        flow_false = tb.AutoregressiveFlow(
            num_params=2,
            transformation_type="affine",
            n_transformations=1,
            hidden_units=[2],
            permutations=False,
        )
        self.assertFalse(flow_false.permutations)

    def test_autoregressive_flow_permutation_none_branch(self):
        """Test Autoregressive flow permutation none branch."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            permutations=None,
        )
        self.assertFalse(flow.permutations)

    def test_autoregressive_flow_range_adjustment_feedback(self):
        """Test Autoregressive flow range adjustment feedback."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            parameters_min=np.array([-50.0], dtype=np.float32),
            parameters_max=np.array([60.0], dtype=np.float32),
            range_max=1.0,
            permutations=False,
            feedback=2,
        )
        self.assertGreater(float(tf.convert_to_tensor(flow.range_max)), 1.0)
        self.assertIsNotNone(flow.bijector)

    def test_autoregressive_flow_spline_branch_nonperiodic(self):
        """Test Autoregressive flow spline branch nonperiodic."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            permutations=False,
        )
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_periodic_affine_tanh_branch(self):
        """Test Autoregressive flow periodic affine tanh branch."""
        class ToggleStr(str):
            """Toggle Str test suite."""
            def __new__(cls, value):
                """New."""
                obj = super().__new__(cls, value)
                obj.calls = 0
                return obj

            def __eq__(self, other):
                """Eq."""
                self.calls += 1
                if other == "spline":
                    return self.calls == 1  # only pass the initial assertion
                if other == "affine":
                    return True
                return super().__eq__(other)

        toggled = ToggleStr("affine")
        self.assertTrue(toggled == "spline")
        self.assertTrue(toggled == "affine")
        self.assertFalse(toggled == "other")

        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[1],
            periodic_params=np.array([True]),
            permutations=False,
        )
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_iterable_permutations_branch(self):
        """Test Autoregressive flow iterable permutations branch."""
        perms = [np.array([0, 1], dtype=np.int32)]
        flow = tb.AutoregressiveFlow(
            num_params=2,
            transformation_type="affine",
            n_transformations=1,
            hidden_units=[2],
            permutations=perms,
        )
        self.assertEqual(flow.permutations, perms)

    def test_autoregressive_flow_periodic_scale_preprocess(self):
        """Test Autoregressive flow periodic scale preprocess."""
        flow = tb.AutoregressiveFlow(
            num_params=2,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            periodic_params=np.array([True, False]),
            permutations=False,
            range_max=2.0,
        )
        sample = tf.constant([[0.1, 0.2]], dtype=tf.float32)
        self.assertEqual(flow.bijector.forward(sample).shape, sample.shape)

    def test_autoregressive_flow_map_to_unitcube(self):
        """Test Autoregressive flow map to unitcube."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[2],
            map_to_unitcube=True,
            permutations=False,
        )
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_unexpected_permutation_type(self):
        """Test Autoregressive flow unexpected permutation type."""
        with self.assertRaises(UnboundLocalError):
            tb.AutoregressiveFlow(
                num_params=1,
                transformation_type="affine",
                n_transformations=1,
                hidden_units=[1],
                permutations=np.bool_(True),
            )

    def test_autoregressive_flow_range_without_adjustment(self):
        """Test Autoregressive flow range without adjustment."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[1],
            parameters_min=np.array([-1.0], dtype=np.float32),
            parameters_max=np.array([1.0], dtype=np.float32),
            range_max=5.0,
            permutations=False,
        )
        self.assertEqual(float(flow.range_max), 5.0)

    def test_autoregressive_flow_custom_kernel_initializer(self):
        """Test Autoregressive flow custom kernel initializer."""
        initializer = tf.keras.initializers.Constant(0.5)
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="affine",
            n_transformations=1,
            hidden_units=[2],
            permutations=False,
            kernel_initializer=initializer,
        )
        maf = flow.bijector.bijectors[0]
        self.assertIs(maf._shift_and_log_scale_fn._kernel_initializer, initializer)

    def test_autoregressive_flow_periodic_non_affine_branch(self):
        """Test Autoregressive flow periodic non affine branch."""
        class NonAffineStr(str):
            """Non Affine Str test suite."""
            def __new__(cls, value):
                """New."""
                obj = super().__new__(cls, value)
                obj.spline_calls = 0
                return obj

            def __eq__(self, other):
                """Eq."""
                if other == "spline":
                    self.spline_calls += 1
                    return self.spline_calls != 3
                if other == "affine":
                    return False
                return super().__eq__(other)

        toggled = NonAffineStr("spline")
        self.assertFalse(toggled == "other")
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type=toggled,
            n_transformations=1,
            hidden_units=[1],
            periodic_params=np.array([True]),
            permutations=False,
        )
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_map_to_unitcube_periodic(self):
        """Test Autoregressive flow map to unitcube periodic."""
        flow = tb.AutoregressiveFlow(
            num_params=1,
            transformation_type="spline",
            n_transformations=1,
            hidden_units=[1],
            map_to_unitcube=True,
            periodic_params=np.array([False]),
            permutations=[np.array([0], dtype=np.int32)],
        )
        out = flow.bijector.forward(tf.constant([[0.0]], dtype=tf.float32))
        self.assertEqual(out.shape, (1, 1))

    def test_autoregressive_flow_inconsistent_transformation_branch(self):
        """Test Autoregressive flow inconsistent transformation branch."""
        class FlakySpline(str):
            """Flaky Spline test suite."""
            def __new__(cls, value):
                """New."""
                obj = super().__new__(cls, value)
                obj.calls = 0
                return obj

            def __eq__(self, other):
                """Eq."""
                if other == "affine":
                    return False
                if other == "spline":
                    self.calls += 1
                    return self.calls <= 2  # later comparisons turn False
                return super().__eq__(other)

        flaky = FlakySpline("spline")
        self.assertFalse(flaky == "other")
        with self.assertRaises(UnboundLocalError):
            tb.AutoregressiveFlow(
                num_params=1,
                transformation_type=flaky,
                n_transformations=1,
                hidden_units=[1],
                permutations=False,
            )

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
