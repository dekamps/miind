/**
\page page_installation Installation

\section sec_tpd Third Party Dependencies

It is necessary to install <a href="http://cern.root.ch">ROOT</a>. ROOT is a powerful analysis platform that has similar capabilities as MATLAB,
but is optimized for high end performance. Under Unix systems, this is straightforward.
For almost all UNIX-like distributions there is a binary. This can be installed in a directory of your choice. If you
have <i>sudo</i> or <i>root</i> permission, you can opt to have ROOT installed under <i>/usr/local</i>, thereby making the framework
available to all users. It is also possible to install the package locally. Regardless of whether you install the package locally or under
<i>/usr/local</i>, the top of the ROOT directory structure is always a directory called 'root'.

Alternatively, you can compile the package from source, using the <i>configure</i> script} in the top directory of the download. There is a comprehensive
description on how to do this: https://root.cern.ch/building-root.  Make sure you have all the prerequisites installed that are listed at https://root.cern.ch/build-prer\
equisites.

Make sure that the version you use is configured with \verbatim --enable-python, --enable-table, --enable-mathmore \endverbatim. You
can use Python to inspect the simulation results, and convert them to numpy objects if you feel the need.

Whether you install ROOT locally or system-wide, make sure that the script <i>root/bin/thisroot.sh</i> is sourced, i.e. issue the command:
\verbatim source ~/root/bin/thisroot.sh \endverbatim if you have installed the package in your home directory. 
You have to do this every time before you use													       
ROOT, so it is worth to include in a <i>.bashrc</i> file or equivalent.

													       
You will also need:
													       
- The GNU Scientific Library, GSL for short
																				    
- A recent (> 1.48) version of BOOST.

section sec_procedure Procedure
Whether you down load the tar file or checkout the code from the repository, you will have a top directory called 'miind-git'. This is the <B> MIIND_ROOT</B>.
Where you place this is immaterial.
Perform the following steps:


 - Directly below 'code', create a directory called 'build'. This 'build' directory will be at the same level as 'apps' and 'libs'.
 - 'cd build'
 - 'ccmake ..'
 - Indicate whether you want a Release or a Debug version, by setting the <B>CMAKE_BUILD_TYPE</B> field.
 - You may have to indicate where ROOT is. CMake is intelligent enough to work out where, once you have provided the location of the root excutable.
 - Configure ('c')
 - Generate the Make file ('g')
 - Quit ('q')
 - Type 'make'
 
The libraries will built in <i>build/libs</i>, the excetuables in buid/apps.
In general, you want set your <B>PATH</B> and <B>PYTHONPATH</B> such that they include the path to <B>MIIND_ROOT</B>/python.						      

CMake can be used directly from the command line, e.g.:						 
								  \verbatim                                      
cmake path_to_miind_src -DENABLE_MPI=TRUE -DCMAKE_BUILD_TYPE=Debug -DMPIEXEC=/opt/local/bin/openmpirun     
         \endverbatim
  To compile the project type:
\verbatim
make
\endverbatim
To build the documentation type:
\verbatim
make doc
\endverbatim



\section sec_ubuntu A Clean Ubuntu Install

We start with a clean Ubuntu 14-04 machine. Install the following packages with <i>sudo apt-get install</i>:
\verbatim
 g++ (this should be at least g++ 4.8, which you should get by default.)
 python-scipy
 cmake-curses-gui
 libboost-all-dev
 libgsl0-dev
  git
\endverbatim

Do not use a Ubuntu package for ROOT! It does exist, but misses a few libraries that MIIND depends on.
  Go to the ROOT web site: http://root.cern.ch, go to <i>Download</i> and click on the most recent version. Download the Ubuntu binary
    and unpack it in a directory of your choice. Issue the command 
\verbatim source cwd/root/bin/thisroot.sh
\endverbatim
, where 'cwd' should be replaced by the name of the
    directory where you unpacked. If you now type 'root' anywhere in a shell, ROOT's CINT interpreter should start and a splash screen should appear.
Exit CINT by typing ' .q'. Start a Python shell, and type 'import ROOT'. This module should now load without any comment. You may want
to incorporate the 'source' command described above in a <i>.bashrc</i> file or equivalent.

*/

