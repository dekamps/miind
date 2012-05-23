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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "SimulationBuilderCode.h"
#include "XMLConnectionCode.h"
#include "XMLNodes.h"
#include "XMLSimulationCode.h"
#include "../PopulistLib/PopulistLib.h"
#include "../PopulistLib/TwoPopulationTest.h"
#include "../DynamicLib/TestDefinitions.h"
#include "../DynamicLib/WilsonCowanExample.h"


using namespace PopulistLib;

namespace MiindLib {
	template <> bool SimulationBuilder<double>::GenerateExample(const string& file_name) const
	{
		ofstream ofst(file_name.c_str());
		if (! ofst){
			cerr << "Can't open file here" << endl;
			return false;
		}

		vector<string> vector_canvas_names;
		vector_canvas_names.push_back("E");
		vector_canvas_names.push_back("I");
		XMLRunParameter par_run_xml("wilson_cowan",true,false,false,vector_canvas_names,DEFAULT_CANVAS);

		// we're not interest in intermediate results
		InactiveReportHandler inactive_handler;
	
		SimulationRunParameter 
			run_parameter
			(
				inactive_handler,
				10000000,
				0.0,
				0.1,
				1e-4,
				1e-4,
				1e-5,
				"test/WilsonCowanAlgorithmTest.log"
			);

		// Create a WilsonCowan parameter for excitatory Nodes
		WilsonCowanParameter 
			excitatory_parameter
			(
				TAU_EXCITATORY,
				F_MAX,
				F_NOISE
			);

		// And for inhibitory Nodes
		WilsonCowanParameter 
			inhibitory_parameter
			(
				TAU_INHIBITORY,
				F_MAX,
				F_NOISE
			);

		// Create the Algorithms to initialize the Nodes
		WilsonCowanAlgorithm 
			excitatory_algorithm
			(
				excitatory_parameter
			);
		excitatory_algorithm.SetName("Wilson-Cowan excitatory Algorithm");

		WilsonCowanAlgorithm 
			inhibitory_algorithm
			(
				inhibitory_parameter
			);
		inhibitory_algorithm.SetName("Wilson-Cowan inhibitory Algorithm");

		// Creat a constant background rate Node, first create the algorithm :
		D_RateAlgorithm 
			rate
			(
				1
			);
		rate.SetName("Background Algorithm");

		XMLSimulation<double>::algorithm_vector vec_alg;
		vec_alg.push_back(XMLSimulation<double>::algorithm_pointer(excitatory_algorithm.Clone()));
		vec_alg.push_back(XMLSimulation<double>::algorithm_pointer(inhibitory_algorithm.Clone()));
		vec_alg.push_back(XMLSimulation<double>::algorithm_pointer(rate.Clone()));

		XMLNode node_e("EXCITATORY","E", "Wilson-Cowan excitatory Algorithm");
		XMLNode node_i("INHIBITORY","I", "Wilson-Cowan inhibitory Algorithm");
		XMLNode node_bg("INHIBITORY","Background","Background Algorithm"); // take care, in the Wilson-Cowan algorithm this is inhibitory
		node_vector vec_nodes;
		vec_nodes.push_back(node_e);
		vec_nodes.push_back(node_i);
		vec_nodes.push_back(node_bg);


		XMLConnection<double> con_J_EE   ("E","E", ALPHA);
		XMLConnection<double> con_J_EI   ("I","E", BETA);
		XMLConnection<double> con_J_IE   ("E","I", GAMMA);
		XMLConnection<double> con_J_II   ("I","I", DELTA);

		XMLConnection<double> con_J_EE_BG("Background","E", ETA);
		XMLConnection<double> con_J_IE_BG("Background","I", ETA);

		typedef vector<XMLConnection<double> > connection_vector;
		connection_vector vec_con;
		vec_con.push_back(con_J_EE);
		vec_con.push_back(con_J_II);
		vec_con.push_back(con_J_IE);
		vec_con.push_back(con_J_EI);
		vec_con.push_back(con_J_EE_BG);
		vec_con.push_back(con_J_IE_BG);

		XMLSimulation <double>
			simulation
			(
				"double",				// connection type
				par_run_xml,			// xml run parameter, specifying file name and output conditions
				run_parameter,			// the simulation run parameter from the two population test
				vec_alg,				// vector containing pointers to all algorithms
				vec_nodes,
				vec_con
			);

		simulation.ToStream(ofst);
		return true;
	}

	template <> bool SimulationBuilder<PopulationConnection>::GenerateExample(const string& file_name) const
	{
		ofstream ofst(file_name.c_str());
		if (! ofst){
			cerr << "Cant't open file here" << endl;
			return false;
		}
		vector<string> vec_canvas_names;
		vec_canvas_names.push_back("LIF E");
		vec_canvas_names.push_back("LIF I");
		CanvasParameter par(0.0,0.05,0.,10.0,0., 20e-3,0.,250.0);
		XMLRunParameter par_run_xml("two_pop",true,true,false,vec_canvas_names,par);
		PopulationAlgorithm alg_e(TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER_POP);
		alg_e.SetName("LIF_excitatory Algorithm");
		PopulationAlgorithm alg_i(TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP);
		alg_i.SetName("LIF_inhibitory Algorithm");
		Pop_RateAlgorithm alg_bg(RATE_TWOPOPULATION_EXCITATORY_BACKGROUND);
		alg_bg.SetName("Cortical Background Algorithm");

		XMLSimulation<PopulationConnection>::algorithm_vector vec_alg;
		vec_alg.push_back(XMLSimulation<PopulationConnection>::algorithm_pointer(alg_e.Clone()));
		vec_alg.push_back(XMLSimulation<PopulationConnection>::algorithm_pointer(alg_i.Clone()));
		vec_alg.push_back(XMLSimulation<PopulationConnection>::algorithm_pointer(alg_bg.Clone()));

		XMLNode node_e("EXCITATORY","LIF E", "LIF_excitatory Algorithm");
		XMLNode node_i("INHIBITORY","LIF I", "LIF_inhibitory Algorithm");
		XMLNode node_bg("EXCITATORY","Cortical Background","Cortical Background Algorithm");
		node_vector vec_nodes;
		vec_nodes.push_back(node_e);
		vec_nodes.push_back(node_i);
		vec_nodes.push_back(node_bg);

		PopulationConnection 	
			connection_J_EE
			(
				TWOPOPULATION_C_E*TWOPOPULATION_FRACTION,
				TWOPOPULATION_J_EE
			);

		PopulationConnection
			connection_J_IE
			(
				static_cast<Number>(TWOPOPULATION_C_E*TWOPOPULATION_FRACTION),
				TWOPOPULATION_J_IE
			);

		PopulationConnection
			connection_J_EI
			(
				TWOPOPULATION_C_I,
				-TWOPOPULATION_J_EI
			);

		PopulationConnection
			connection_J_II
			(
				TWOPOPULATION_C_I,
				-TWOPOPULATION_J_II
			);

		PopulationConnection
		connection_J_EE_BG
			(
				TWOPOPULATION_C_E*(1-TWOPOPULATION_FRACTION), 
				TWOPOPULATION_J_EE
			);

		PopulationConnection
		connection_J_IE_BG
			(
				static_cast<Number>(TWOPOPULATION_C_E*(1 - TWOPOPULATION_FRACTION)),
				TWOPOPULATION_J_IE
			);

		XMLConnection<PopulationConnection> con_J_EE   ("LIF E","LIF E", connection_J_EE);
		XMLConnection<PopulationConnection> con_J_II   ("LIF I","LIF I", connection_J_II);
		XMLConnection<PopulationConnection> con_J_IE   ("LIF E","LIF I", connection_J_IE);
		XMLConnection<PopulationConnection> con_J_EI   ("LIF I","LIF E", connection_J_EI);

		XMLConnection<PopulationConnection> con_J_EE_BG("Cortical Background","LIF E", connection_J_EE_BG);
		XMLConnection<PopulationConnection> con_J_IE_BG("Cortical Background","LIF I", connection_J_IE_BG);

		typedef vector<XMLConnection<PopulationConnection> > connection_vector;
		connection_vector vec_con;
		vec_con.push_back(con_J_EE);
		vec_con.push_back(con_J_II);
		vec_con.push_back(con_J_IE);
		vec_con.push_back(con_J_EI);
		vec_con.push_back(con_J_EE_BG);
		vec_con.push_back(con_J_IE_BG);

		XMLSimulation <PopulationConnection>
			simulation
			(
				"Pop",				// connection type
				par_run_xml,		// xml run parameter, specifying file name and output conditions
				TWOPOP_PARAMETER,	// the simulation run parameter from the two population test
				vec_alg,			// vector containing pointers to all algorithms
				vec_nodes,
				vec_con
			);

		simulation.ToStream(ofst);

		return true;
	}
}