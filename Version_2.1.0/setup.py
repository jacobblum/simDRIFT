from setuptools import setup

setup(
    name='dmri-MCSIM',
    version='2.1.0',    
    description='testing',
    url='https://github.com/jacobblum/dMRI-MCSIM/tree/dev',
    author='J.S. Blum, K.L. Utt',
    author_email='jacobblum@wustl.edu, klutt@wustl.edu',
    license='???',
    packages=['src'],
    install_requires=['cupy',
                      'numpy',
                      'numba',
                      'nibabel'                     
                      ],

)