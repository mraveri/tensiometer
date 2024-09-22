============
Tensiometer
============
-------------------------------
 Test a model until it breaks!
-------------------------------
:Tensiometer: A tool to test the level of agreement/disagreement between high dimensional posterior distributions.
:Author: Marco Raveri and Cyrille Doux
:Homepage: https://tensiometer.readthedocs.io
:Source: https://github.com/mraveri/tensiometer
:References: `2105.03324 <https://arxiv.org/abs/2105.03324>`_ (non-Gaussian tension metrics), 
             `1806.04649 <https://arxiv.org/abs/1806.04649>`_ and 
             `1912.04880 <https://arxiv.org/abs/1912.04880>`_ (Gaussian tension metrics),
             `2112.05737 <https://arxiv.org/abs/2112.05737>`_ (measured parameters).

.. image:: https://github.com/mraveri/tensiometer/actions/workflows/test.yml/badge.svg
    :target: https://github.com/mraveri/tensiometer/actions/workflows/test.yml
.. image:: https://readthedocs.org/projects/tensiometer/badge/?version=latest
    :target: https://tensiometer.readthedocs.org/en/latest
.. image:: https://coveralls.io/repos/github/mraveri/tensiometer/badge.svg?branch=master
    :target: https://coveralls.io/github/mraveri/tensiometer?branch=master
.. image:: https://img.shields.io/pypi/v/tensiometer.svg?style=flat
    :target: https://pypi.python.org/pypi/tensiometer/

Description
============

The tensiometer package is a collection of tools to help understanding the structure of high 
dimensional posterior distributions. 
In particular it implements a number of metrics to quantify the level of agreement/disagreement
between different distributions.
Some of these methods are based on a `Gaussian approximation <https://arxiv.org/abs/1806.04649>`_.
Others are capable of capturing `non-Gaussian features <https://arxiv.org/abs/2105.03324>`_ of the distributions 
thanks to machine learning techniques.

The best way to get up to speed is to walk through some worked example, based on 
what is needed:

* `Normalizing flow models for posterior distributions <https://tensiometer.readthedocs.org/en/latest/example_synthetic_probability.html>`_;
* `Tension between two Gaussian posteriors <https://tensiometer.readthedocs.org/en/latest/example_gaussian_tension.html>`_;
* `Tension between two non-Gaussian posteriors <https://tensiometer.readthedocs.org/en/latest/example_non_gaussian_tension.html>`_;
* `Posterior profiles <https://tensiometer.readthedocs.org/en/latest/example_posterior_profiles.html>`_;
* `Measured parameters <https://tensiometer.readthedocs.org/en/latest/example_measured_parameters.html>`_;
* `Quantify convergence of chains <https://tensiometer.readthedocs.org/en/latest/example_chains_convergence_test.html>`_;


Installation
=============

The tensiometer package is available on PyPI and can be easily installed with::

  pip install tensiometer

Alternatively one can download the source code from github::

  git clone https://github.com/mraveri/tensiometer.git

and install it locally with the shortcut::

  make install

You can test that the code is working properly by using::

  make test

Dependencies
=============

Tensiometer uses mostly standard python packages.
Notable exceptions are GetDist, Tensorflow and Tensorflow Probability.
Installing the last two is likely painful and we advice to not delegate that to 
automatic dependency resolvers...

For the full list of requirements see the `requirements.txt` file.

Testing the code
================

Tensiometer has a suite of unit tests to make sure everything is properly installed. 
This is especially useful considering that tensiometer relies on a number of external libraries.

To run all tests give the command::

  make test

To run tests and get coverage statistics first make sure that the package `coverage <https://pypi.org/project/coverage/>`_ is installed. 
Then you can use the command::

  make test_with_coverage

As you can see coverage is not complete, pull requests improving test coverage are most welcome.

Documentation
=============

The documentation is automatically built for each release on `readthedocs <https://tensiometer.readthedocs.io/en/latest/>`_.

If you want to build locally the documentation you shoud install `sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_.
Then you can install the readthedocs documentation format with::

  pip install sphinx-rtd-theme

Then you can issue the command::

  make documentation

Acknowledgements
================

We thank Samuel Goldstein, Shivam Pandey for help developing the code.

****************