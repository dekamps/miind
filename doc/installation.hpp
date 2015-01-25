/**
\page installation Installation
\section sec_prelim Preliminaries. 
At the moment, it is necessary to install 
<a href="http://root.cern.ch">ROOT</a>. Under Unix systems, this is 
straightforward. ROOT is a powerful analysis platform that has similar
capablities as MATLAB, but is optimized for high performance. Use
a pre-compiled version or compile yourself. Make sure that the
version you use is configured with --enable-python. You can
use  Python to inspect the simulation results, and convert
them to numpy objects if you feel the need. 




You will also need:
<ul>
 <li> The Gnu Scientific Library, <a href="http://www.gnu.org/software/gsl/">GSL</a> for short</li>
 <li>  A recent (> 1.48) version of <a href="http://www.boost.org">BOOST</a>.
</ul>

\section Basic Installation
Whether you down load the tar file or checkout the code from the
repository, you will have a top directory called 'code'. Where you place
this is immaterial. Preform the following steps:
<ol>
<li> Directly below 'code', create a directory called 'build'. This 'build' directory will be at the same level as 'apps' and 'libs'.</li>
<li> cd build </li>
<li> ccmake ..</li>
<li> Indicate whether you want a Release or a Debug version.</li>
<li> You may have to indicate where ROOT is. CMake is intelligent enough
to work out where, once you have provided the location of the root excutable.</li>
<li> Generate the Make file ('g')</li> 
<li> Quit ('q') </li>
<li> Type 'make'</li>
</ol>
The libraries will built in build/libs, the excetuables in buid/apps.

\section Release
Once you have done this and want to make changes, it may be easier
to run cmake, although the description below applies to ccmake as well.
<ol>
<li> Run cmake:

You have the following options:
<dl>
<dt>
ENABLE_TESTING [Default: TRUE]
</dt>
<dd>compiles the tests</dd>
<dt>
ENABLE_MEMORY_TEST [Default: FALSE]
</dt>
<dd>compiles the memory tests. Attention this increase the duration of the test run significantly</dd>
<dt>
ENABLE_COVERAGE [Default: FALSE]
</dt>
<dd>allows to generate a coverage report. Does not work at the moment</dd>
<dt>
DEBUGLEVEL [Default: logINFO]
</dt>
<dd>set the debug level see the avilable levels at \ref provided_debug_levels</dd>
<dt>
ENABLE_MPI [Default: FALSE]
</dt>
<dd>enable mpi for the miind mpi lib</dd>
<dt>
MPIEXEC
</dt>
<dd>the path to the mpirun executable</dd>
<dt>
CMAKE_INSTALL_PREFIX [Default: /usr/local]
</dt>
<dd>allows to specify the installation path</dd>
<dt>
CMAKE_BUILD_TYPE
</dt>
<dd>controls the type of build
<ul><li>None (CMAKE_C_FLAGS or CMAKE_CXX_FLAGS used)
</li><li>Debug (CMAKE_C_FLAGS_DEBUG or CMAKE_CXX_FLAGS_DEBUG)
</li><li>Release (CMAKE_C_FLAGS_RELEASE or CMAKE_CXX_FLAGS_RELEASE)
</li><li>RelWithDebInfo (CMAKE_C_FLAGS_RELWITHDEBINFO or CMAKE_CXX_FLAGS_RELWITHDEBINFO
</li><li>MinSizeRel (CMAKE_C_FLAGS_MINSIZEREL or CMAKE_CXX_FLAGS_MINSIZEREL)
</li></ul>
</dd>
</dl>

e.g. run cmake like this

\code cmake path_to_miind_src -DENABLE_MPI=TRUE -DCMAKE_BUILD_TYPE=Debug -DMPIEXEC=/opt/local/bin/openmpirun\endcode
</li>
<li> To compile the project run:

\code make \endcode
</li>
<li> To build the docs:

\code make doc \endcode
</li>
</ol>

\section Testing
For testing you can run the included test and measure the coverage of the tests.
<ol>
<li> Call cmake with ENABLE_TESTING true. If you want a coverage report also set ENABLE_COVERAGE to true and make a debug build.
\code \code cmake path_to_miind_src -DENABLE_COVERAGE=TRUE -DENABLE_TESTING=TRUE -DCMAKE_BUILD_TYPE=Debug \endcode
</li>
<li> To execute the tests run:

\code make test \endcode
</li>
<li> To generate the coverage report run:
\code make coverage \endcode
</li>
</ol>

*/
