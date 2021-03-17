============
Installation
============

Install MIIND package with pip
------------------------------

To install with pip::

    $ python -m pip install miind
	
MIIND is available on Windows, MacOS, and Linux for python versions >=3.6.
	
Standalone package
------------------

`MIIND with CUDA Support <https://github.com/dekamps/miind/blob/master/package/miind_1.06-1_all_cuda.deb>`_

`MIIND without CUDA Support <https://github.com/dekamps/miind/blob/master/package/miind_1.06-1_all.deb>`_

Additional python libraries which need to be installed using pip or conda:

- numpy
- matplotlib
- shapely
- descartes
- scipy

Standalone Docker
-----------------
Pull MIIND from DockerHub::

    $ docker pull hughosborne/miind:latest

CUDA is currently disabled for the MIIND Docker image.

Building Python MIIND From Source
---------------------------------
Build and Install Python MIIND Locally::

    $ python setup.py install

Python MIIND depends on:

- Boost
- GSL
- Freeglut
- OpenGL
- FFTW
- PugiXML
- Python3-Dev (Python.h)

Python MIIND optionally depends on:

- CUDA Toolkit
- OpenMP
- MPI
- ROOT

On Windows, vcpkg is used for building Python MIIND therefore only CUDA drivers and Ninja are required in addition to cmake and a compiler.

Building Standalone MIIND From Source
-------------------------------------

Standalone MIIND can also be built in the tranditional way (create a build directory and run cmake then install).

Additional python libraries which need to be installed using pip or conda:

- numpy
- matplotlib
- shapely
- descartes
- scipy

Set the following environment variables:

- OMP_NUM_THREADS (See OpenMP documentation)
- Add <MIIND_Installation_Directory>/share/miind/python to PATH
- Add <MIIND_Installation_Directory>/share/miind/python to PYTHONPATH