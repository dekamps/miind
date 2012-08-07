/**
\page dependencies Dependencies

<ol>
<li>compilers: see \ref compilers</li>
<li>root: see \ref root</li>
<li>gsl: see http://www.gnu.org/software/gsl/</li>
<li>boost mpi: \ref boost</li>
</ol>


\section compilers Compiler
You need to have a compiler which supports C++11. For example gcc 4.7 or clang 3.1 or later.
You also need to have a mpi compiler like openmpi for compiling the mpi code.

\section root Root
Root needs to be installed with the following flags
\code
./configure --enable-table --enable-explicitlink
make
source /bin/thisroot.sh
\endcode
for more details see http://root.cern.ch/drupal/

\section boost Boost
\code
./bootstrap.sh
\endcode
add the following code to project-config.jam:
\code
using mpi ;
\endcode
and build boost
\code
./b2
./b2 install
\endcode
for more detail see http://www.boost.org/
*/
