/*!
\mainpage Miind Documentation
\section sec_intro Introduction
MIIND provides a simulation framework for neural simulations. It focusses on population level
descriptions of neural dynamics, i.e. it does not provide a siumulator for individual neurons,
but models populations directly. It provides support for simple neural mass models, but
focusses strongly on so-called population density techniques.

To get a feeling for the simulator and its capabilities,
we encourage you to read the following <a href="http://www.sciencedirect.com/science/article/pii/S0893608008001457">paper</a> 
(an open access <a href="http://eprints.whiterose.ac.uk/5235/">preprint</a> version is also available). 

\image html hbp.png

Currently, part of the developent is funded by the <a href="http://www.humanbrainproject.eu/">Human Brain Project</a>.  
A new official release date will be 1 October 2014. The new release will contain many new algorithms, including
other populations than leaky-integrate-and-fire ones. The documentation will be revamped.

\section sec_download Download
The latest tar bal can be found <a href="http://sourceforge.net/projects/miind">here</a>. Install on Unix platforms
is straightforward, using cmake. The procedure is explained
in \ref installation. You can checkout a snapshot of the latest
code with:
cvs -d :pserver:anonymous@miind.cvs.sf.net:/cvsroot/miind co -P code
This will change soon, though.

The components are Open Source software under a BSD licence.
\section licence Licence

MIIND is free and Open Source. It is distributed under a modified BSD license (see \ref license)

\section sec_install Installation
Use cmake and make. For more details see \ref installation.

\section sec_dependencies Dependencies
For more details see \ref dependencies


\section sec_wiki WIKI
The MIIND <a href="http://sourceforge.net/p/miind/wiki/Home">wiki</a>.
The WIKI contains more details about the extra work that needs to be done to install MIIND under windows.
*/

