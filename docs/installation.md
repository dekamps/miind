---
title: 'Installation'
---

# Python Package [linux/mac/windows] {: #python-package}

`python -m pip install miind`

The easiest way to get MIIND is through python pip. MIIND is available for Windows, Linux, and MacOS for Python >= 3.6.

# Standalone Package [linux] {: #standalone-package}

[*MIIND with CUDA Support*](https://github.com/dekamps/miind/blob/master/package/miind_1.06-1_all_cuda.deb)

[*MIIND without CUDA Support*](https://github.com/dekamps/miind/blob/master/package/miind_1.06-1_all.deb)

Additional python libraries which need to be installed using pip or conda:

numpy
matplotlib
shapely
descartes
scipy

# Standalone Docker {: #docker}

`docker pull hughosborne/miind:latest`

CUDA is currently disabled for the MIIND Docker image.

# Building Python MIIND From Source [linux/mac/windows] {: #building-miind-python}

`python setup.py install`

Python MIIND depends on:

Boost
GSL
Freeglut
OpenGL
FFTW
PugiXML
Python3-Dev (Python.h)

Python MIIND optionally depends on:

CUDA Toolkit
OpenMP
MPI
ROOT

On Windows, vcpkg is used for building Python MIIND therefore only CUDA drivers and Ninja are required in addition to cmake and a compiler.

# Building Standalone MIIND From Source [linux/mac/windows] {: #building-miind-standalone}

Standalone MIIND can also be built in the tranditional way (create a build directory and run cmake then install).

Additional python libraries which need to be installed using pip or conda:

numpy
matplotlib
shapely
descartes
scipy

Set the following environment variables:

OMP_NUM_THREADS (See OpenMP documentation)
Add <MIIND_Installation_Directory>/share/miind/python to PATH
Add <MIIND_Installation_Directory>/share/miind/python to PYTHONPATH