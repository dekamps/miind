/*!
\mainpage Miind Documentation
\section sec_announce Announcement
MIIND on the Hackathon session of the HBP summit. There is a <a href="http://miind.sf.net/tutorial.pdf">tutorial</a>.
Although very rough, it will help you get going.

MIIND version 1.02 was released on 19 September 2015. It contains a major bug fix.

MIIND version 1.01 was released on 11 March 2015. Appart from bug fixes it contains a visualization tool
to monitor progress of a running simulation (including the evolution of the densities!).



MIIND version 1.0 was released 25 January 2015. It is now solely dedicated to population density techniques 
and neural mass models. Important new features include:
- Full support for 1D neural models. These include leaky-integrate-and-fire and quadratic-integrate-and-fire neurons. Other neuron  models
can easily be provided or implemented.
- Support for arbitrary large synaptic efficacies: Fokker-Planck equations are contained as a special case.
- MPI support

Many new features are in developement. These include:

- Support for multi-dimensional models
- Support for non Poisson statistics
- Support for activity dependent efficacies

The old neural network code, including a HMAX implementation is still available, but will no longer be maintained.



\section sec_intro Introduction
MIIND provides a simulation framework for neural simulations. It focusses on population level
descriptions of neural dynamics, i.e. it does not provide a simulator for individual neurons,
but models population activity directly. It provides support for simple neural mass models, but
focusses strongly on so-called population density techniques.


To get a feeling for the simulator and its capabilities, go to the \subpage page_examples page.

To run the simulator, go to the \ref workflow page, after installation is completed. Make sure
you have had a glance at the \ref ugh page.

\image html hbp.png
Currently, part of the developent is funded by the <a href="http://www.humanbrainproject.eu/">Human Brain Project</a>.  

\section sec_download Download
The latest tar bal can be found <a href="http://sourceforge.net/projects/miind">here</a>. Install on Unix platforms
is straightforward, using cmake. The procedure is explained
in \ref installation. You can checkout a snapshot of the latest
code with:
git clone git://git.code.sf.net/p/miind/git miind-git

Make sure you use the integration branch. We will change to the main branch in February.

The components are Open Source software under a BSD licence.
\section licence Licence

MIIND is free and Open Source. It is distributed under a  BSD license (see \ref license)

\section sec_install Installation
Use cmake and make. For more details see \ref installation.

\section sec_dependencies Dependencies
For more details see \ref dependencies


\section sec_wiki WIKI
The MIIND <a href="http://sourceforge.net/p/miind/wiki/Home">wiki</a>.
The WIKI contains more details about the extra work that needs to be done to install MIIND under windows.

\section sec_bug Report a Bug
Please create a <a href="https://sourceforge.net/p/miind/bugs/">ticket</a>.

*/

