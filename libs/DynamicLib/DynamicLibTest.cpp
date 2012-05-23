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
#ifdef WIN32
#pragma warning(disable: 4996 4800) // these warnings come from ROOT header files
#endif

#include "DynamicLibTest.h"
#include <valarray>
#include <functional>
#include <fstream>
#include <sstream>
#include <TFile.h>
#include <TGraph.h>
#include <TPostScript.h>
#include <TSVG.h>
#include <TH2F.h>
#include <TROOT.h>
#include <TApplication.h>
#include <TStyle.h>
#include <TCanvas.h>
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "../SparseImplementationLib/SparseImplementationTest.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../NumtoolsLib/NumtoolsTest.h"
#include "AlgorithmBuilder.h"
#include "AsciiReportHandler.h"
#include "DynamicLibException.h"
#include "DynamicNode.h"
#include "DynamicNetworkCode.h"
#include "TestDefinitions.h"
#include "RootHighThroughputHandler.h"
#include "WilsonCowanExample.h"

using namespace std;
using namespace SparseImplementationLib;
using namespace DynamicLib;

DynamicLibTest::DynamicLibTest(boost::shared_ptr<ostream> p_stream):
LogStream(p_stream)
{
}

DynamicLibTest::~DynamicLibTest()
{
}

bool DynamicLibTest::Execute() 
{
/*	if ( ! GridAndStateStreamingTest() )
		return false;
	Record("GridAndStateStreamingTest succeeded");

	if ( ! WilsonCowanAlgorithmTest() )
		return false;
	Record("WilsonCowanAlgorithmTest succeeded");

	if ( ! WilsonCowanNetworkTest() )
		return false;
	Record("WilsonCowanNetworkTest succeeded");

	if (! WilsonCowanExampleTest() )
		return false;
	Record("WilsonCowanExampleTest succeeded");

	if ( ! InnerProductTest() )
		return false;
	Record("InnerProductTest succeeded");

	if ( ! NetworkStreamingTest() )
		return false;
	Record("NetworkStreamingTest succeeded");

	if ( ! RootHandlerTest() )
		return false;
	Record("RootHandlertest succeeded");

	if ( ! MultipleRootTest() )
		return false;
	Record("MultipleRootTest succeeded");
*/
	if (! RootHighThroughputHandlerTest() )
		return false;
	Record("RootHighTroughputHandlerTest succeeded");

	if (! MultipleHighThroughputTest() )
		return false;
	Record("RootHighTroughputHandlerTest succeeded");

/*
	if ( ! NetworkCopyTest() )
		return false;
	Record("NetworkCopyTest succeeded");

	if ( ! MaxNumberIterationsTest() )
		return false;
	Record("MaxNumberIterationsTest succeeded");

	if (! SpatialNetworkTest() )
		return false;
	Record("SpatialNetworkTest succeeded");

	if (! WilsonCowanTest() )
		return false;
	Record("WilsonCowanTest succeeded");

	if (!this->SimRunParSerializeTest() )
		return false;
	Record("SimRunParSerializeTest succeeded");

	if (!this->RateAlgorithmSerializeTest() )
		return false;
	Record("RateAlgorithmSerializeTest succeeded");

	if (!this->BuildRateAlgorithm() )
		return false;
	Record("BuildrateAlgorithm");

	if (!this->WilsonCowanSerializeTest() )
		return false;
	Record("WilsonCowanSerializeTest succeeded");

	if (!this->BuildWilsonCowanAlgorithm() )
		return false;
	Record("BuildWilsonCowanAlgorithm succeeded");

	if (!this->CanvasParameterWriteTest() )
		return false;
	Record("CanvasParameterWriteTest succeeded");
*/
	return true;
}

bool DynamicLibTest::WilsonCowanAlgorithmTest() const
{
	const Time     time_begin  = 0;
	const Time     time_end    = 5;
	const TimeStep time_h      = 1e-4;
	const TimeStep time_report = 1e-2;
	const TimeStep time_update = 1e-2;
	const Number   NUMBER_ITERATIONS = 1000000;

	vector<double> grid_vector(1,0);
	AlgorithmGrid grid(grid_vector);

	WilsonCowanParameter 
		parameter
		(
			TIME_MEMBRANE,
			F_MAX,
			F_NOISE,
			0
		);

	WilsonCowanAlgorithm algorithm(parameter);
	D_Afferent afferent(AFFERENT);



	// we're not interest in intermediate results
	InactiveReportHandler inactive_handler;
	SimulationRunParameter run_parameter
				(
					inactive_handler,
					NUMBER_ITERATIONS,
					time_begin,
					time_end,
					time_h,
					time_report,
					time_update,
					"test/WilsonCowanAlgorithmTest.log"
				);

	algorithm.Configure(run_parameter);

	for (Time time = time_begin + time_h; time < time_end; time += time_h)
		algorithm.EvolveNodeState
		(
			afferent.begin(),
			afferent.end(),
			time
		);
	
	double f_result = algorithm.State()[0];

	if (! NumtoolsLib::IsApproximatelyEqualTo(f_result,F_RESULT,EPSILON))
		return false;

	return true;
};

bool DynamicLibTest::InnerProductTest() const
{
	D_DynamicNode node1;
	D_DynamicNode node2;
	D_DynamicNode node3;

	node1.SetValue(1.0);
	node2.SetValue(1.0);
	node3.SetValue(1.0);

	typedef pair<D_AbstractSparseNode*,double> connection; 
	typedef D_AbstractSparseNode::predecessor_iterator predecessor_iterator;
	vector<connection> vector_of_connections;

	vector_of_connections.push_back(connection(&node1,1));
	vector_of_connections.push_back(connection(&node2,2));
	vector_of_connections.push_back(connection(&node3,3));

	// Need a concrete Algorithm to calculate the inner product

	WilsonCowanParameter parameter
				(
					0,
					0,
					0,
					0
				);

	WilsonCowanAlgorithm algorithm(parameter);

	// Note the cumbersome notation
	// This is necessary for testing purposes because this is a 'naked' algorithm:
	// it doesn't belong to a Node, normally the Node provides the PredecessorIterators

	double f_inner_product = 
		algorithm.InnerProduct
		(
			predecessor_iterator(&(*vector_of_connections.begin())),
			predecessor_iterator(&(*vector_of_connections.begin()) + vector_of_connections.size())
		);

	if (
		! NumtoolsLib::IsApproximatelyEqualTo
			(
				f_inner_product,
				6.0,
				1e-10
			) 
		)
		return false;

	return true;
}

D_DynamicNetwork DynamicLibTest::CreateWilsonCowanNetwork() const
{

	// code moved to separate source file 
	// doxygen example purposes
	D_DynamicNetwork network = WilsonCowanExample();
	return network;
}

bool DynamicLibTest::WilsonCowanNetworkTest() const
{

	D_DynamicNetwork network = CreateWilsonCowanNetwork();

	const string report_path_name = STRING_TEST_DIR + STRING_ASCIIREPORTNAME;

	AsciiReportHandler handler(report_path_name);

	const SimulationRunParameter 
		Wilson_Cowan_Simulation_Run_Parameter
		(
			handler,
			100000,		// not more than 10000 iterations expected
			T_START,	// Start at 0 ms
			T_END,		// End time: 50 ms,
			1e-4,		// Report,
			1e-3,		// Update
			1e-5,		// Reasonable time scale for the network to evolve
			"test/WilsonCowanNetworkTest.log"
		);
	bool b_configure = network.ConfigureSimulation
				(
					Wilson_Cowan_Simulation_Run_Parameter
				);

	if (! b_configure )
		return false;


	bool b_evolve = network.Evolve();


	return b_evolve;
}

bool DynamicLibTest::NetworkStreamingTest() const
{

	D_DynamicNetwork network = CreateWilsonCowanNetwork();

	string path = STRING_TEST_DIR + STRING_WILSONCOWAN_FILE;

	ofstream stream_out(path.c_str());
	if ( ! stream_out )
		throw DynamicLibException(STR_OPENNETWORKFILE_FAILED);

	network.ToStream(stream_out);
	return true;
}

bool DynamicLibTest::GridAndStateStreamingTest() const
{
	vector<double> bla(3);
	bla[0] = 0;
	bla[1] = 1;
	bla[2] = 2;

	NodeState state(bla);

	string path_state = STRING_TEST_DIR + STRING_NODESTATE_STREAMING_TEST;

	ofstream stream_state(path_state.c_str());
	if ( ! stream_state )
		return false;
	if (! state.ToStream(stream_state) )
		return false;

	stream_state.flush();

	ifstream stream_input_state(path_state.c_str());
	if (! stream_input_state )
		return false;


	NodeState state_input(vector<double>(10,0));
	if (! state_input.FromStream(stream_input_state))
		return false;

	if (
			state_input[0] != 0 ||
			state_input[1] != 1 ||
			state_input[2] != 2
		)
		return false;

	AlgorithmGrid grid(bla);

	string path_grid = STRING_TEST_DIR + STRING_GRID_STREAMING_TEST;
	ofstream stream_grid(path_grid.c_str());
	if (! stream_grid )
		return false;
	if (! grid.ToStream(stream_grid))
		return false;

	stream_grid.flush();

	ifstream stream_input_grid(path_grid.c_str());
	if (! stream_input_grid)
		return false;

	AlgorithmGrid grid_input(vector<double>(10,0));

	if ( ! grid_input.FromStream(stream_input_grid))
		return false;

	// is the order correct ?

	if (
			grid_input.ToStateVector()[0] != 0 ||
			grid_input.ToStateVector()[1] != 1 ||
			grid_input.ToStateVector()[2] != 2
		)
		return false;

	return true;
}

bool DynamicLibTest::RootHandlerTest() const
{
	D_DynamicNetwork network = CreateWilsonCowanNetwork();

	string path = STRING_TEST_DIR + STRING_WILSONCOWAN_ROOTFILE;
	RootReportHandler handler(path, false, true);
	handler.AddNodeToCanvas(NodeId(1));
	handler.AddNodeToCanvas(NodeId(2));

	const SimulationRunParameter 
		Wilson_Cowan_Simulation_Run_Parameter
		(
			handler,
			100000,		// not more than 10000 iteration expected
			T_START,	// Start at 0 ms
			2e-3,		// End time: 50 ms,
			1e-3,		// Report,
			1e-2,		// Update
			1e-5,		// Reasonable time scale for the network to evolve
			"test/roothandlertest.log"
		);
	bool b_configure = 
		network.ConfigureSimulation
		(
			Wilson_Cowan_Simulation_Run_Parameter
		);

	if (! b_configure )
		return false;


	bool b_evolve = network.Evolve();
	return b_evolve;

}

bool DynamicLibTest::RootHighThroughputHandlerTest() const 
{
	D_DynamicNetwork network = CreateWilsonCowanNetwork();

	string path = STRING_TEST_DIR + STRING_HIGHTHROUGHPUT_ROOTFILE;
	RootHighThroughputHandler handler(path, true, true);

	const SimulationRunParameter 
		Wilson_Cowan_Simulation_Run_Parameter
		(
			handler,
			100000,		// not more than 10000 iteration expected
			T_START,	// Start at 0 ms
			2e-3,		// End time: 50 ms,
			1e-3,		// Report,
			1e-2,		// Update
			1e-5,		// Reasonable time scale for the network to evolve
			"test/roothighthroughputhandlertest.log"
		);
	bool b_configure = 
		network.ConfigureSimulation
		(
			Wilson_Cowan_Simulation_Run_Parameter
		);

	if (! b_configure )
		return false;


	bool b_evolve = network.Evolve();
	return b_evolve;

	return true;
}

bool DynamicLibTest::MultipleHighThroughputTest() const
{
	D_DynamicNetwork network = CreateWilsonCowanNetwork();

	string path = STRING_TEST_DIR + "MULTIPLE" + STRING_HIGHTHROUGHPUT_ROOTFILE;
	RootHighThroughputHandler handler(path, true, true);

	const SimulationRunParameter 
		Wilson_Cowan_Simulation_Run_Parameter
		(
			handler,
			100000,		// not more than 10000 iteration expected
			T_START,	// Start at 0 ms
			2e-3,		// End time: 50 ms,
			1e-3,		// Report,
			1e-2,		// Update
			1e-5,		// Reasonable time scale for the network to evolve
			"test/roothighthroughputhandlertest.log"
		);
	bool b_configure = 
		network.ConfigureSimulation
		(
			Wilson_Cowan_Simulation_Run_Parameter
		);

	if (! b_configure )
		return false;


	bool b_evolve = network.Evolve();

	// Nor run again on the same file. It should overwrite the old one.

	b_configure = 
		network.ConfigureSimulation
		(
			Wilson_Cowan_Simulation_Run_Parameter
		);

	if (! b_configure )
		return false;

	b_evolve = network.Evolve();

	string path2 = STRING_TEST_DIR + "MULTIPLE2" + STRING_HIGHTHROUGHPUT_ROOTFILE;
	RootHighThroughputHandler handler2(path2, true, true);

	const SimulationRunParameter 
		Wilson_Cowan_Simulation_Run_Parameter2
		(
			handler,
			100000,		// not more than 10000 iteration expected
			T_START,	// Start at 0 ms
			2e-3,		// End time: 50 ms,
			1e-3,		// Report,
			1e-2,		// Update
			1e-5,		// Reasonable time scale for the network to evolve
			"test/roothighthroughputhandlertest2.log"
		);

	b_configure = 
		network.ConfigureSimulation
		(
			Wilson_Cowan_Simulation_Run_Parameter2
		);

	if (! b_configure )
		return false;


	b_evolve = network.Evolve();

	return true;
}

bool DynamicLibTest::NetworkCopyTest() const
{
	return true;

}

bool DynamicLibTest::MaxNumberIterationsTest() const
{
	D_DynamicNetwork network = CreateWilsonCowanNetwork();

	// not interested in results just want iteration exception in log file
	InactiveReportHandler handler;


	const SimulationRunParameter 
		Wilson_Cowan_Simulation_Run_Parameter
		(
			handler,
			1,		// Deliberately ridiculous small value
			T_START,	// Start at 0 ms
			T_END,		// End time: 50 ms,
			1e-4,		// Report,
			1e-2,		// Update,
			1e-5,		// Reasonable time scale for the network to evolve,
			string("test/iteratorexception.log")
		);
	bool b_configure = 
		network.ConfigureSimulation
		(
			Wilson_Cowan_Simulation_Run_Parameter
		);

	if (! b_configure )
		return false;


	bool b_evolve = network.Evolve();

	// result should be false
	return (! b_evolve);
}

bool DynamicLibTest::MultipleRootTest() const
{
	return true;
}

bool DynamicLibTest::SpatialNetworkTest() const
{
	D_RateAlgorithm alg(0.0);
	D_DynamicNetwork net;

	NodeId id = net.AddNode(alg, EXCITATORY);


	SpatialPosition spat(0.0, 1.0, 2.0);

	// this should work without problem
	bool b_ret = net.AssociateNodePosition(id, spat);

	if ( ! b_ret)
		return false;

	return true;
}

// This is another WilsonCowan example, which is exported by Doxygen
// to the website.
bool DynamicLibTest::WilsonCowanTest() const
{

	// define a D_Network, a network whose weights are doubles
	D_DynamicNetwork network_wctest;

	Time tau = 10e-3; //10 ms 
	Rate rate_max = 100.0;
	double noise = 1.0;

	// define some efficacy
	Efficacy epsilon = 1.0;

	// define some input rate
	Rate nu = 0;

	// Define a node with a fixed output rate
	D_RateAlgorithm rate_alg(nu);
	NodeId id_rate = network_wctest.AddNode(rate_alg,EXCITATORY);

	// Define the receiving node 
	WilsonCowanParameter par_sigmoid(tau,rate_max,noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);
	NodeId id = network_wctest.AddNode(algorithm_exc,EXCITATORY);

	// connect the two nodes
	network_wctest.MakeFirstInputOfSecond(id_rate,id,epsilon);


	bool b_configure = network_wctest.ConfigureSimulation(PAR_WILSONCOWAN);

	if (! b_configure)
		return false;

	bool b_evolve = network_wctest.Evolve();
	if (! b_evolve)
		return false;
	
	// ending
	return true;
}


bool DynamicLibTest::SimRunParSerializeTest() const
{
	ofstream ostr("test/simrun.par");

	if (! ostr){
		cout << "Opening test file failed" << endl;
		return false;
	}

	PAR_WILSONCOWAN.ToStream(ostr);
	ostr.close();

	ifstream ist("test/simrun.par");

	SimulationRunParameter par_run(WILSONCOWAN_HANDLER,ist);

	return true;
}

bool DynamicLibTest::RateAlgorithmSerializeTest() const
{
	RateAlgorithm<double> alg(1000);
	alg.SetName("harige gorilla");

	ofstream ofst("test/rate.alg");
	if (! ofst)
		return false;

	alg.ToStream(ofst);
	ofst.close();

	ifstream ifst("test/rate.alg");
	RateAlgorithm<double> algin(ifst);

	ofstream ofst2("test/newrate.alg");
	algin.ToStream(ofst2);

	return true;
}

bool DynamicLibTest::BuildRateAlgorithm() const
{
	ifstream st("test/rate.alg");

	AlgorithmBuilder<double> build;
	boost::shared_ptr< AbstractAlgorithm<double> > p_alg = build.Build(st);

	return true;
}

bool DynamicLibTest::WilsonCowanSerializeTest() const
{
	ofstream ofst("test/wilsoncowan.alg");
	WilsonCowanParameter par;
	par._f_input = 1.54;
	par._f_noise = 1.01;
	par._time_membrane = 10e-3;
	par._rate_maximum = 503.0;

	WilsonCowanAlgorithm alg(par);
	alg.SetName("treurige gebeurtenis");

	alg.ToStream(ofst);
	ofst.close();

	ifstream ifst("test/wilsoncowan.alg");
	WilsonCowanAlgorithm algin(ifst);

	ofstream ofst2("test/newwilsoncowan.alg");
	algin.ToStream(ofst2);

	return true;
}

bool DynamicLibTest::CanvasParameterWriteTest() const 
{
	ofstream ofst("test/canvas.par");

	CanvasParameter par(0.0, 1.0, 0, 20, -1.0, 10.0, 2.0, 5.0 );
	par.ToStream(ofst);
	ofst.close();

	ifstream ifst("test/canvas.par");

	CanvasParameter par_in;
	par_in.FromStream(ifst);

	ofstream ofst2("test/canvas2.par");
	par_in.ToStream(ofst2);

	return true;
}

bool DynamicLibTest::BuildWilsonCowanAlgorithm() const
{
	ifstream st("test/wilsoncowan.alg");

	AlgorithmBuilder<double> build;
	boost::shared_ptr< AbstractAlgorithm<double> > p_alg = build.Build(st);

	return true;
}

bool DynamicLibTest::WilsonCowanExampleTest() const
{

	D_DynamicNetwork net = DynamicLib::WilsonCowanExample();

	ofstream ofst("test/wilsoncowanexample.net");
	ofst << net;

	RootReportHandler handler("test/wilsoncowanexample.root",false,false);
	handler.AddNodeToCanvas(NodeId(1));
	handler.AddNodeToCanvas(NodeId(2));

	SimulationRunParameter
		par_run
		(
			handler,
			10000000,
			0.0,
			0.1,
			1e-4,
			1e-4,
			1e-5,
			"log"
		);

	net.ConfigureSimulation(par_run);
	net.Evolve();

	return true;
}
namespace { 
	void Graph1(){
		// example 1: coupled excitatory-inhibitory population with inhibitory input
		TFile file("test/wilsoncowanexample.root");

		TGraph* p_exc = (TGraph*)(file.Get("rate_1")); //yes! naked pointers, due to the peculiairities of ROOT's ownership model
		TGraph* p_inh = (TGraph*)(file.Get("rate_2"));

		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);

		TH2F hist("h","Population rates",500,0,0.1,500,0,15.0);
		hist.SetXTitle("t (s)");
		hist.SetYTitle("spikes/s");
		TCanvas c1;
		TSVG myps("test/dynamiclib_test_wilsoncowandouble.svg");
		myps.Range(15,12);
		hist.Draw();
		p_exc->Draw("L");
		p_inh->Draw("L");

		myps.Close();
		file.Close();
	}

	void Graph2(){


		TFile file2("test/wilsonresponse.root");
		TGraph* p_gr = (TGraph*)file2.Get("rate_2"); // this is the exciatory population

		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);


		TH2F hist2("h","Population rate",500,0,0.1,500,0,60.0);
		hist2.SetXTitle("t (s)");
		hist2.SetYTitle("spikes/s");
		TCanvas c1;

		TSVG your_ps("test/dynamiclib_test_wilsonresponse.svg");

		your_ps.Range(15,12);
		hist2.Draw();
		p_gr->Draw("L");

		your_ps.Close();
		file2.Close();
	}
}

void DynamicLibTest::ProcessResults()
{
	// Doxygen expects that PNG versions of the images produces here are in the images directory on sourceforge.
	//
	// This function writes two wilson cowan examples, first a complex one, then a simple, in the documentation (website) the order of presentation  is reversed.
	// The order is immaterial, doxygen only knows abou the file names.
	Graph1();

	// example 2: now a single population, demonstrating convergens to half its maximum firing rate.
	Graph2();
}