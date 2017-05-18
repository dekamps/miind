/**
\page page_workflow Workflow
\section intro_workflow Introduction

There are two work flows:
    - Adapting an XML file, using a Python interpreter to produce automatically generated C++ code, compiling and analysing the results in
         Python, using PyROOT.
    - Create and Run a C++ Program , compiling and linking to the MIIND libraries; running the simulation and analysing the results in Python,
         using PyROOT.

\subsection workflow_python Python Workflow
In the first case you will not have to deal with C++ directly and even experienced C++ programmers may find the XML files easier to handle than
the corresponding C++ programs. It is certainly recommended for a first try, even if you intend to use C++.

We suggest that you install  MIIND in a local directory. MIIND is small. The directory containing the miind code is called the <B>MIIND_ROOT</B>. If you have worked
through the installation successfully
(see \ref page_installation), you will have a top directory for miind in the location where you installed it, and withing this directory an \a apps,
\a libs, \a examples  and a \a python directory. You will also have created a  \a build directory yourself. Upon successful compilation, the \a build directory itself will
 contain an \a apps  directory, that contains a \a BasicDemos directory that contains executables that directly correspond to the source files in the
\a apps/BasicDemos directory under  the <B>MIIND_ROOT</B>.


Now create a working directory. It is recommended that you create this directory outside the MIIND installation, ideally in a place where
you have back up. Now copy the three files present in \a <B>MIIND_ROOT</B>/example/single_lif  to your working directory.
Then give the command: \verbatim miind.py --d single response.xml \endverbatim. The following now happens:

  - A new directory \a single is created in <B>MIIND_ROOT</B>/apps
  - In this directory a C++ file is created that is the C++ code corresponding to the XML file.
  - The building environment (CMake) is adapted to incorporate this new executable into the building process. Your new executable
    will be  a full fledged part of MIIND now. At this stage is has not been compiled yet.

You could simple type \verbatim make \endverbatim  in the \a <B>MIIND_ROOT</B>/build directory and you would see that the new program will be compiled.
It is easier to type: \verbatim submit.py single \endverbatim. Note that 'single' is the name we used in the first step. You can choose your another name
but must use it consistently, as this name will be part of MIIND's build structure. This command achieves the following:

  -  The new source file is compiled and linked to the MIIND libraries creating a new executable in <B>MIIND_ROOT</B>/build/apps/single,
called 'response'.
  - This executable will be run.
  - A directory \a single will be created in \a <B>MIIND_ROOT</B>/jobs. In this directory the simulation results will be present. There will
    also be a file \a joblist that contains a list of all simulation results. Having this list is a predictable location makes it very easy
    to write analysis programs that operate on multiple jobs and parameter sweeps.


By issuing the command \verbatim  miind.py --r --d single \endverbatim the \a single directory is removed from the build chain. Neither the executables,
nor the simulation results are removed by this command; you have to do this by hand. This entails removing the directory \a single from
\a <B>MIIND_ROOT</B>/apps, \a <B>MIIND_ROOT</B>/build/apps and \a <B>MIIND_ROOT</B>/jobs.


\subsection sub_sec_batch Submitting Batch Jobs

Parameter sweeps tend to result in a large number of executables that you want to run in parallel. There is an alternative to \verbatim submit.py \endverbatim 
that works much the same,
but rather than running the executables in sequence, \verbatim launch.py \endverbatim  submits batch jobs for each executable. Unlike \a submit.py, \a launch.py does not
use a python script to execute the newly created executables, but it calls a python script which in turns calls a shell script that effectuates a batch submission of the
executable. The shell script is in \a <B>MIIND_ROOT</B>/python and is called \a submit.sh. This shell script is particular to our HPC cluster, and will
need adapting to your own HPC environment. It is unlikely to work straight out of the box, but we are willing to help, if you can provide details on the submission
system on your HPC cluster.

\section sec_analysis Analyzing the results

Running the miind executable will produce a \a .root file. You can analyse the results conveniently in Python, using either the ROOT objects directly,
or converting them into numpy objects that can be analysed in numpy, scipy and visualized with Matplotlib.

Copy the root file to a directory of your choice. Make sure that the <B>PYTHONPATH</B> variable is set to pick up the <B>ROOT</B> module.

Open a Python shell in your favouring Python environment and type:
\verbatim
import ROOT
\endverbatim

If this does not generate errors, you are in business. Now open the file:
\verbatim
f=ROOT.TFile('yourfile.root')
\endverbatim
You can inspect the file:

\verbatim
f.ls()
\endverbatim
You will get a list of names of TGraph objects. To get them from the file, do, for example:

\verbatim
g=f.Get('rate_1')
g.Draw('AP')
\endverbatim
A canvas with the firing rate graph should now pop up.



ROOT is a very powerful analysis environment, with more extensive visualization capabilities than Matplotlib, but nothing prevents you from using SciPy for your analysis.
Consider a ROOT.TGraph object with name 'data':
\code{.py}
x_buff = data.GetX()
y_buff = data.GetY()
N = data.GetN()
x_buff.SetSize(N)
y_buff.SetSize(N)
# Create numpy arrays from buffers, copy to prevent data loss
x_arr = np.array(x_buff,copy=True)
y_arr = np.array(y_buff,copy=True)
\endcode
Here is a brief introduction to PyROOT: https://www-zeuthen.desy.de/~middell/public/pyroot/pyroot.html.


\section sec_adv_cpp Advanced C++ Workflow
Again, consider the MIIND directory structure:

\image html org.pdf
\image latex org.pdf "Each directory has a file CMakeLists.txt. It is straightforward to add new executables by adding these files." width=10cm

If you are comfortable with C/C++ it may well be possible that you want to write your own code based on the MIIND API. It is relatively straightforward to
create your own simulations by copying existing code and adding the copies as new files into the compilation tree. MIIND uses <a href="http://www.cmake.org">CMake</a>,
and in each directory of Fig. \ref{fig-cmake} you will find a directory <i>CMakeLists.txt</i>. Consider the directory <i>BasicDemos</i> under  <B>MIIND_ROOT</B>/apps.
It contains the file <i>population-example.cpp</i>. Copy this file to <i>new.cpp</i>. Now edit the file <i>CMakeLists.txt</i> in the <i>BasicDemos</i> directory. Under the
section <i># executables</i> you will find entries such as the following:
\verbatim
add_executable(populationDemo population-example.cpp)
target_link_libraries(populationDemo \${LIBLIST})
\endverbatim
Copy this entry in the same file under the same section  using an editor and change this entry into:
\verbatim}
add_executable(myProgram new.cpp)
target_link_libraries(myProgram \${LIBLIST})
\endverbatim
This is sufficient to add the new program to the build structure. Typing \verbatim make \endverbatim in the build directory will cause a new Makefile to be generated, which subsequently will
cause the <i>new.cpp</i> to be compiled and linked into a new executable. Of course this program will do exactly the same thing as <i>population-example.cpp</i>, until you start
making modifications to the code. When you do this, the full MIIND API is at your disposal. Earlier in the <i>CMakeLists.txt</i> you will find entries that ensure that your code
will find MIIND's header files:

\verbatim
include_directories( ../../libs/MPILib )
include_directories( ../../libs/GeomLib )

include_directories( \${GSL_INCLUDE_DIRS} )
link_directories( \${GSL_LIBRARY_DIRS} )
include_directories( \${ROOT_INCLUDE_DIRS} )
link_directories( \${ROOT_LIBRARY_DIRS} )
\endverbatim
*/
