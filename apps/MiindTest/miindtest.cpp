// Copyright (c) 2005 - 2011 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifdef WIN32
#pragma warning(disable: 4996)
#endif

#include <iostream>
#include <ClamLib/ClamLibTest.h>
#include <ConnectionismLib/ConnectionismTest.h>
#include <DynamicLib/DynamicLibTest.h>
#include <NetLib/NetLibTest.h>
#include <NumtoolsLib/NumtoolsTest.h>
#include <PopulistLib/TestPopulist.h>
#include <SparseImplementationLib/SparseImplementationTest.h>
#include <StructnetLib/StructnetLibTest.h>
#include <UtilLib/UtilTest.h>
#include <LayerMappingLib/Test.h>

using ClamLib::ClamLibTest;
using ConnectionismLib::ConnectionismTest;
using DynamicLib::DynamicLibTest;
using NetLib::NetLibTest;
using NumtoolsLib::NumtoolsTest;
using PopulistLib::TestPopulist;
using SparseImplementationLib::SparseImplementationTest;
using StructnetLib::StructNetLibTest;
using UtilLib::UtilTest;

using std::endl;

/*! 
 * \mainpage A general overview of MIIND 
 * \section overhaul  MIIND's documentation is being overhauled at the moment. 
 * Please contact us at M.deKamps@leeds.ac.uk if you are curious and consider using it. We can probably set you up quickly.
 *	\section sec_miind_intro An introduction to MIIND
 *
 * These are the following ways of using MIIND:
 * <ol>
 * <li> \subpage simpage. Study neuronal circuits at the population level and analyse results with Python, MATLAB, ROOT or in C++.</li>
 * <li> \subpage modelpage Configure your own simulations and analyse the results with Python, MATLAB or ROOT.</li>
 * <li> \subpage cpluspluspage for simulating network processes. Program your own node process, but use MIIND the miind network framework for network representation. Copying networks, visualisation and serialisation are handled by the framework.</li>
 * </ol>
 * \section download_and_install Download and Install MIIND
 * Instructions for downloading and installing MIIND can be found at the 
 *  <a href="https://sourceforge.net/apps/mediawiki/miind/index.php?title=Main_Page">MIIND WIKI</a>.
 *
 * \section  license License
 *
 * MIIND is free and Open Source. It is distributed under a modified BSD license (see \subpage license_page), where the only extra condition is that if you use MIIND for a scientific publication you mist cite the
 * \subpage currently_valid_reference.
 *
 * \section  track_record Track Record
 * MIIND was used for work leading to the following publications:
 * <ul>
 * <li>D. G. Harrison and M. De Kamps, (2011). A dynamical model of feature-based attention with strong lateral inhibition to resolve competition among candidate feature locations. In Proceedings of the AISB 2011 Symposium on
 * Architectures for Active Vision, April 2011.</li>
 * <li>D. G. Harrison and M. De Kamps. A dynamical neural simulation of featurebased attention and binding in a recurrent model of the ventral stream. In press. (Presented at 12th Neural Computation and Psychology Workshop
 * March 2010, to be published by World Scientific Press).</li>
 * <li>Marc de Kamps, Volker Baier, Johannes Drever, Melanie Dietz, Lorenz Mosenlechner, Frank van der Velde (2008). <a href="http://www.sciencedirect.com/science/article/pii/S0893608008001457">The state of MIIND</a>, Neural Networks, Volume 21, Issue 8, Pages 1164-1181, ISSN 0893-6080, DOI: 10.1016/j.neunet.2008.07.006.</li>
 * <li>Frank van der Velde and  Marc de Kamps, (2007). A neural model of global visual saliency. In S. Vosniadou, D. Kayser and A. Protopapas (eds), Proceedings of the European Cognitive Science Conference 2007, pp 383-388. New York, Lawrence Erlbaum.</li>
 * <li>Marc de Kamps and Volker Baier, Multiple Interacting Instantiations of Neuronal Dynamics (MIIND): a Library for Rapid Prototyping of Models in Cognitive Neuroscience,Proceedings of IJCNN2007, 2007, Florida</li>
 * <li>Frank van der Velde and Marc de Kamps, Neural Blackboard Architectures of Combinatorial Structures in Cognition, Behavioral and Brain Sciences (2006), 29, 37-70</li>
 * <li>Frank van der Velde and Marc de Kamps, From Neural Dynamics to True Combinatorial Structures, Behavioral and Brain Sciences (2006), 29, 88-108</li>
 * <li>Marc de Kamps and Frank van der Velde, Neural Blackboard Architectures: the Realization of Compositionality and Systematicity in Neural Networks, Journal of Neural Engineering (2006), 3, R1-R12</li>
 * <li>Marc de Kamps, An Analytic Solution of the Reentrant Poisson Master Equation and its Application in the Simulation of Large Groups of Spiking Neurons, Proceedings of IJCNN2006 (WCCI2006), 2006, Vancouver</li>
 * <li>Marc de Kamps, A Simple and Stable Numerical Solution for the Population Density Equation, Neural computation, 2003, 15(9), 2129-2146</li>
 * <li>Frank van der Velde and Marc de Kamps, From Knowing What to Knowing Where: Modeling Object-Based Attention with Feedback Disinhibition of Activation, Journal of Cognitive Neuroscience, (2001), 13(4), 479-491</li>
 * <li>Marc de Kamps and Frank van der Velde, From Artificial Neural Networks to Spiking Neuron Populations and Back Again, Neural Networks, (2001), 14(6-7), 941-953</li>
 * </ul>
 
 *
 * \page license_page The License for MIIND
 * The license text is produce below, starting with the Copyright line. Different source files may be copyrighted by different people. For much of MIIND
 * you need the Gnu Scientific Library(http://www.gnu.org/software/gsl/), which is distributed GNU General Public License, which is mainly relevant
 * if you ever aim to develop commercial applications based on MIIND. 
 *
 * Copyright (c) 2005 - 2011 Marc de Kamps
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * <ul>
 *    <li>Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.</li>
 *    <li>Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
 *      and/or other materials provided with the distribution.</li>
 *    <li>Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
 *      without specific prior written permission</li>
 * </ul>
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *      If you use this software in work leading to a scientific publication, you should include a reference there to
 *      the 'currently valid reference', which can be found at http://miind.sourceforge.net
 *
 * \page currently_valid_reference Currently valid reference for MIIND
 * If you use MIIND for work leading to a scientific population, you must the currenly valid reference for MIIND, which at present is:
 * Marc de Kamps, Volker Baier, Johannes Drever, Melanie Dietz, Lorenz Mosenlechner, Frank van der Velde (2008). The state of MIIND, Neural Networks, Volume 21, Issue 8, Pages 1164-1181, ISSN 0893-6080, DOI: 10.1016/j.neunet.2008.07.006.
 * (http://www.sciencedirect.com/science/article/pii/S0893608008001457)
 *
 * Older literature on MIIND:
 *
 *
 * \page simpage  MIIND as a neural simulator
 * \section simpage_advantage Advantages
 * MIIND as a neural simulator operates on the population level. It does not simulate individual neurons, other neural simulators, such as NEST, NEURON, GENESIS
 * or BRIAN  can be used for that. MIIND simulates on the population level, which means that certain characteristics of the population are simulated directly. Perhaps
 * the most famous example is provided by Wilson-Cowan equations <a href="doi:10.1016/S0006-3495(72)86068-5">doi:10.1016/S0006-3495(72)86068-5</a>, which model a neuron  population's firing rate in response to 
 * external input. Population level modelling has a number of advantages:
 * <ul>
 * <li>It is often much more computationally efficient to simulate at the population level</li>
 * <li>It can simplify simulations dramatically: complex networks of cortical circuits are difficult to define in most neural simulators. On the population level
 * this is much more easily done</li>
 * <li>Once a simulation on the individual level is run, often statistics on the population level must be collected. Population level modelling bypasses this step.</li>
 * </ul>
 * \section simpage_usage Usage
 * 
 * The following methods are available for  running MIIND as a neural simulator.
 * <ol>
 * <li>\subpage running_as_xml. This requires no explicit coding and can be a useful way for running MIIND without programming experience. Networks of considerable
 * complexity can now be simulated with a wide variety of algorithms using just an XML editor (you need to have the MIIND software installed, obviously).
 * Go to the section  to find out more.</li>
 *  <li>\subpage running_as_cplusplus. This section will explain how the various simulation techniques can be started using MIIND's C++ API. Examples
 *  included are Wilson Cowan dynamics (\ref wilson_cowan), steady state population rates (techniques employed by  by Amit and Brunel, for example),
 *  and full population dynamics, which in includes Fokker-Planck equations to model leaky-integrate-and fire dynamics as a special case.
 * </li>
 * <li>As code, from a Python script.</li>
 * * </ol>
 * \page running_as_xml Running MIIND from an XML configuration file.
 *	\section xml_introduction Introduction
 *
 * We will look at various examples of increasing complexity. First you need to locate the miind executable. Run the executable as follows:
 * './miind --g bla.xml --w double'. This will generate an XML file. It contains a simple example simulation and can be used immediately to test
 * the set up: if you run the command './miind bla.xml', the progam runs briefly and produces a file with extension .root, which contains
 * the simulation results. How you then inspect the simulation results is explained in section \ref inspecting_root_file. Under Windows
 * you have to run from a CMD window using almost identical commands (replace './miind' simply by 'miind'). If the ROOT file
 * is generated your simulation runs correctly.
 *
 *  The general structure of the XML file is explained in \subpage xml_structure . A simple example, setting up a simulation using Wilson-Cowan
 * dynamics is given in \ref xml_wilson_cowan_example. A more complex example, showing the use of population density techniques is given 
 * in \ref xml_population_example. 
 *
 * \section inspecting_root_file Inspecting the simulation results
 * The simulation results are stored in a file with extenstion .root. In this section we will explain how you can have a quick peek
 * at the simulation results, to see if they are conform expectations and how you can convert them into MATLAB files, or access them
 *  with a Python script or C++ program.
 *
 * \page running_as_cplusplus Running MIIND from a C++ program
 * \section cplusplusstarted Getting started
 * Obviously, you need to have MIIND installed. The install and download page tells you how to install MIIND and how to configure the installation so that,
 * presumably, you know where the libraries and include files are. You can then use your favourite environment to link the MIIND framework against your own code.
 * It is worth considering, however, to add your program to the CMake environment and start your own private MIIND, so to speak.
 * \section cplusplus_example_wilsoncowan A simple example of Wilson-Cowan equations.
 * In '\subpage wilson_cowan' a full example of Wilson Cowan equations is discussed, and it is shown how to set up C++ code to simulate
 * simple networks. More complicated examples will suggest themselves, or consider the ClamLib documentation for hierarchies
 * of networks of populations.
 *
 * \section cplusplus_example_popdens A simple example of using population density techniques (Fokker-Planck-type equations).
 * In '\subpage population_algorithm', the modelling of a population of leaky-integrate-and-fire (LIF) neurons in response to varying
 * input is demonstrated with the help of an example program.
 * \section cplusplus_example_workmem A simple example of a working memory model.
 *
 * \page modelpage MIIND as modelling tool for high level cognitive modelling.
 *
 * This will be documented as soon as the ClamVis library will become available.
 * 
 * \page cpluspluspage MIIND as a C++ framework
 * 
 *  MIIND offers support for:
 * 
 * <ol>
 * <li>Sparse network representations. If you need to represent sparse networks, use \subpage SparseImplementationLib. It allows you to create networks, to add nodes, to
 * copy networks safely, and takes care of serialization (i.e. writing to and reading from disk). Because node and edge types (NodeType and WeightType, respectively)
 * are template parameters, the networks can be customized to your own user-defined types.</li>
 * <li>Dynamic network simulations. Often, in biology and economy processes can be modelled as network processes. The processes taking place at each node can
 * require a considerable amount of implementation and testing. After a thorough testing of the node processes, simulating network processes seems relatively
 * straightforward, but defining networks, serialisation and visualisation of the entire network still can require considerable effort. \subpage DynamicLib offers a generic
 * simulation for sparse network processes. It offers a simulation loop, network creation and modification methods, serialisation and visualisation. The implementor
 * only needs to encapsulate the node process in a so-called AbstractAlgorithm, and to specify network connections. The framework takes care of the rest.</li>
 * <li> Neural dynamics is a special case of a biological network process. \subpage PopulistLib offers network simulations based on Wilson-Cowan dynamics, or
 * population density techniques, such as Fokker-Planck-like equations. MIIND as a neural simulator is based on \ref PopulistLib. It is different from other
 * neural simulators in that it models at the population level, i.e. not at the level of indivdual neurons. This saves simulation time and reduces the complexity
 * of simulations, often without affecting the biological plausibility.</li> 
 * </ol> 
 * Other useful libraries are \subpage ConnectionismLib, \subpage LayerMappingLib and \subpage StructnetLib.
 */


int main(int argc, char* argv[])
{
	cout << "Executing MIIND test suite" << endl;
	boost::shared_ptr<ostream> p(new ofstream("test.log"));
	try {
		if (argc == 1)
		{
		  ofstream test("test/bla");
		  if (! test) {
		    cout << "Please create a directory called 'test' here." << endl;
		    return true;
		  }
			cout << "No arguments specified, testing the whole shebang and logging to cout" << endl;

			{
				LayerMappingLib::Test test( cout );
				if( ! test.Execute() )
				{
					cout << "LayerMappingLib::Test failed." << endl;
					exit( 1 );
				}
			}

			SparseImplementationTest test_sparse(p);
			if ( ! test_sparse.Execute() )
			{
			        cout << "SparseImplementationLibTest failed" << endl;
				exit(1);
			}

			ConnectionismTest test_con(p);
			if ( ! test_con.Execute() )
			{
				cout << "ConnectionismTest failed" << endl;
				exit(1);
			}


			NetLibTest test_net(p);
			if (! test_net.Execute() )
			{
				cout << "NetLibTest failed" << endl;
				exit(1);
			}

			NumtoolsTest test_num(p);
			if (! test_num.Execute() )
			{
				cout << "NumtoolsTest failed" << endl;
				exit(1);
			}

			DynamicLibTest test_dyn(p);
			if (! test_dyn.Execute() )
			{
				cout << "DynamicLibTest failed" << endl;
				exit(1);
			}

 			TestPopulist test_pop(p);
 			if (! test_pop.Execute() )
 			{
 				cout << "TestPopulist failed" << endl;
 				exit(1);
 			}

			StructNetLibTest test_struc(p);
			if ( ! test_struc.Execute() )
			{
				cout << "StructnetLib failed" << endl;
				exit(1);
			}

			UtilTest test_util(p);
			if ( ! test_util.Execute() )
			{
				cout << "UtilTest failed" << endl;
				exit(1);
			}

 			ClamLibTest test_clam(p);
 			if (! test_clam.Execute() )
 			{
 				cout << "ClamLibTest failed" << endl;
 				exit(1);
			}
	
			cout << "miindtest succeeded" << endl;
		}
	}
	catch (UtilLib::GeneralException& excep)
	{
		cout << excep.Description() << endl;
	}
	catch (...)
	{
		cout << "Unknown exception" << endl;
	}

	return 0;
}

