from setuptools import setup
#!/usr/bin/env python

import os
import setuptools



install_requires  = ['numba', 'numpy', 'nibabel']
setuptools.setup(
    name='simDRIFT',
    version='0.0.1',
    description='A software package for forward simulating diffusion weighted MRI',
    url='https://github.com/jacobblum/dMRI-MCSIM',
    author='Jacob Blum, Kainen Utt',
    license='BSD (3-Clause)',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['simDRIFT=src.cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)

