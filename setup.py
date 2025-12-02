import os
from setuptools import setup, find_packages

setup(
    name="dataprocessing",
    version="0.2.0",
    author="Rainette Engbers",
    author_email="rainette.engbers@epfl.ch",
    description="Antarctic eddy covariance and Monin-Obukhov flux data processing toolkit",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    url="https://github.com/Rainette9/DataProcessingScripts",
    packages=['ec', 'mo', 'spc', 'plotting', 'utils'],
    package_dir={
        'ec': 'src/ec',
        'mo': 'src/mo',
        'spc': 'src/spc',
        'plotting': 'src/plotting',
        'utils': 'src/utils',
    },
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'scipy>=1.7.0',
        'cmcrameri>=1.4',
        'windrose>=1.6',
        'metpy>=1.0',  # For meteorological calculations
        'xarray>=0.19.0',  # For multi-dimensional data handling
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0.0',
            'ipython>=7.0.0',
            'pytest>=6.0.0',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords='eddy-covariance, monin-obukhov, atmospheric-science, antarctica, turbulence, flux',
)