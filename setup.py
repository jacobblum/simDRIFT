#!/usr/bin/env python

import os
import setuptools

install_requires  = ['numpy==1.22.4', 'scipy==1.7.2', 'nibabel==3.2.1', 'dipy', 'pytest==7.3.1', 'matplotlib==3.7.2']
setuptools.setup(
    name='simDRIFT',
    description='A software package for forward simulating diffusion weighted MRI',
    url='https://github.com/jacobblum/simDRIFT',
    author='Jacob Blum, Kainen L. Utt',
    license='BSD (3-Clause)',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['simDRIFT=master_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)

