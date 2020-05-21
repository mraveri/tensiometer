============
Tensiometer
============
-------------------------------
 Test a model until it breaks!
-------------------------------
:Tensiometer: utilities to understand concordance and discordance of posterior distributions
:Author: Marco Raveri
:Homepage: https://tensiometer.readthedocs.io
:Source: https://github.com/mraveri/tensiometer
:Reference: mostly https://arxiv.org/abs/1806.04649 and https://arxiv.org/abs/1912.04880

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

The tensiometer package is a collection of tools that extends `GetDist <https://pypi.org/project/GetDist/>`_ capabilities
to test the level of agreement/disagreement between different posterior distributions.

The best way to get up to speed is to read through the worked example
`full worked example <https://tensiometer.readthedocs.org/en/latest/tension_example.html>`_
that you can `run online <https://mybinder.org/v2/gh/mraveri/tensiometer/master?filepath=docs%2Fexample_notebooks%2Ftension_example.ipynb>`_!

Dependencies
=============
* Python 3.6+
* matplotlib 2.2+ (3.1+ recommended)
* scipy
* GetDist
