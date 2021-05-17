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

setup.py defines the cmake options for building MIIND in the variable cmake_args. When using setup.py to build MIIND, these options should be changed if a different configuration is required to the default (ENABLE_OPENMP, ENABLE_CUDA, ENABLE_TESTING). Note that platform specific versions of cmake_args are defined later in the script.

.. code-block:: python
   :caption: cmake-args
   :name: cmake-args
   
	cmake_args = (
		[
			'-DCMAKE_BUILD_TYPE=Release',
			'-DENABLE_OPENMP:BOOL=ON',
			'-DENABLE_MPI:BOOL=OFF',
			'-DENABLE_TESTING:BOOL=ON',
			'-DENABLE_CUDA:BOOL=ON',
			'-DENABLE_ROOT:BOOL=OFF',
			'-DCMAKE_CUDA_FLAGS=--generate-code=arch=compute_30,code=[compute_30,sm_30]'
		]
	)
		
For example, to build MIIND with CUDA disabled and ROOT enabled.

.. code-block:: python
   :caption: cmake-args with CUDA disabled and ROOT enabled
   :name: cmake-args2
   
	cmake_args = (
		[
			'-DCMAKE_BUILD_TYPE=Release',
			'-DENABLE_OPENMP:BOOL=ON',
			'-DENABLE_MPI:BOOL=OFF',
			'-DENABLE_TESTING:BOOL=ON',
			'-DENABLE_CUDA:BOOL=OFF',
			'-DENABLE_ROOT:BOOL=ON'
		]
	)

On Windows, vcpkg is used for building Python MIIND therefore only CUDA drivers and Ninja are required in addition to cmake and a compiler.

Building Standalone MIIND From Source
-------------------------------------

Standalone MIIND can also be built in the tranditional way (create a build directory and run cmake then install).

Create a build directory in the MIIND root directory::

    $ mkdir build
	
Change directory::

	$ cd build
	
Run ccmake to set the required cmake options and generate a cmake file::

    $ cmake ..
	
Once generated, call make install (with admin permissions if required)::

    $ make install

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