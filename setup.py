from setuptools import setup, find_packages

setup(
    name="dataprocessing",
    version="0.1.0",
    author="Rainette Engbers",
    author_email="rainette.engbers@epfl.ch",
    description="Antarctic turbulence and flux data processing toolkit",
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
    ],
)