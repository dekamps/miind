/**
\page installation Installation

\section Release

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
