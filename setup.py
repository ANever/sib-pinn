'''
Covasim installation. Requirements are listed in requirements.txt. There are two
options:
    pip install -e .       # Standard install, does not include optional libraries
    pip install -e .[full] # Full install, including optional libraries
'''

import os
import sys
import runpy
from setuptools import setup, find_packages

# Load requirements from txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'sibpinn', 'version.py')
version = runpy.run_path(versionpath)['__version__']


setup(
    name="sibpinn",
    version=version,
    author="Neverov Andrei",
    description="PINN form Siberia",
    keywords=["PINN","PDE","collocations"],
    platforms=["OS Independent"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'full':  [
            'numpy',
            'matplotlib',
        ],
    }
)
