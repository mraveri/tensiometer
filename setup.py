#!/usr/bin/env python
import re
import os
import sys
import setuptools


# warn against python 2
if sys.version_info[0] == 2:
    print('tensiometer does not support Python 2, \
           please upgrade to Python 3')
    sys.exit(1)


# version control:
def find_version():
    version_file = open(os.path.join(os.path.dirname(__file__),
                                     'tensiometer/__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


# long description (parse readme):
def get_long_description():
    with open('README.rst',  encoding='utf-8-sig') as f:
        lines = f.readlines()
        i = -1
        while '=====' not in lines[i]:
            i -= 1
        return ''.join(lines[:i])


# get requirements:
def get_requirements():
    with open('README.rst',  encoding='utf-8-sig') as f:
        lines = f.readlines()
        i = -1
        while '=====' not in lines[i]:
            i -= 1
        return ''.join(lines[:i])


# setup:
setuptools.setup(name='tensiometer',
                 version=find_version(),
                 description='Tension tools for posterior distributions',
                 long_description=get_long_description(),
                 author='Marco Raveri',
                 url='https://tensiometer.readthedocs.io',
                 license='GPL',
                 project_urls={
                    'Source': 'https://github.com/mraveri/tensiometer',
                    'Tracker': 'https://github.com/mraveri/tensiometer/issues',
                    'Reference': 'https://arxiv.org/abs/1806.04649',
                    'Licensing': 'https://raw.githubusercontent.com/mraveri/tensiometer/master/LICENSE'
                    },
                 packages=setuptools.find_packages(),
                 platforms='any',
                 install_requires=[
                    'GetDist (>=1.1.2)',
                    'numpy',
                    'matplotlib (>=2.2.0)',
                    'scipy (>=1.0.0)',
                    'joblib',
                    'coverage',
                    'tqdm',
                    ],
                 classifiers=[
                    'Development Status :: 2 - Pre-Alpha',
                    'Operating System :: OS Independent',
                    'Intended Audience :: Science/Research',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.6',
                    'Programming Language :: Python :: 3.7',
                    'Programming Language :: Python :: 3.8',
                    ],
                 python_requires='>=3.6',
                 zip_safe=False,
                 keywords=['MCMC']
                 )
