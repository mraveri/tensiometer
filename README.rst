============
Tensiometer
============
-------------------------------
 Test a model until it breaks!
-------------------------------
:Tensiometer: utilities to understand concordance and discordance of posterior distributions
:Author: Marco Raveri and Cyrille Doux
:Homepage: https://tensiometer.readthedocs.io
:Source: https://github.com/mraveri/tensiometer
:References: https://arxiv.org/abs/2105.03324 (non-Gaussian metrics), https://arxiv.org/abs/1806.04649 and https://arxiv.org/abs/1912.04880 (Gaussian)

.. image:: https://travis-ci.org/mraveri/tensiometer.svg?branch=master
    :target: https://travis-ci.org/mraveri/tensiometer
.. image:: https://readthedocs.org/projects/tensiometer/badge/?version=latest
   :target: https://tensiometer.readthedocs.org/en/latest
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/mraveri/tensiometer/master?filepath=docs%2Fexample_notebooks%2Ftension_example.ipynb
.. image:: https://coveralls.io/repos/github/mraveri/tensiometer/badge.svg?branch=master
   :target: https://coveralls.io/github/mraveri/tensiometer?branch=master
.. image:: https://img.shields.io/pypi/v/tensiometer.svg?style=flat
   :target: https://pypi.python.org/pypi/tensiometer/

Description
============

The tensiometer package is a collection of tools to test the level of
agreement/disagreement between different posterior distributions.

The best way to get up to speed is to read through the worked example
`full worked example <https://tensiometer.readthedocs.org/en/latest/tension_example.html>`_
that you can `run online <https://mybinder.org/v2/gh/mraveri/tensiometer/master?filepath=docs%2Fexample_notebooks%2Ftension_example.ipynb>`_! There's also a documented example of
`non-Gaussian tension estimates between DES Y1 and Planck 18 <https://tensiometer.readthedocs.io/en/latest/non_gaussian_tension.html>`_.


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
Notable exceptions are GetDist and Tensorflow.

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
