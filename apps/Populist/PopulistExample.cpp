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

#include <PopulistLib/PopulistLib.h>
#include <MiindLib/MiindLib.h>

using DynamicLib::EXCITATORY_BURST;
using DynamicLib::RootReportHandler;
using DynamicLib::SimulationRunParameter;
using NetLib::NodeId;
using PopulistLib::Efficacy;
using PopulistLib::IntegralRateComputation;
using PopulistLib::InitialDensityParameter;
using PopulistLib::InterpolationRebinner;
using PopulistLib::PopulationAlgorithm;
using PopulistLib::Pop_Network;
using PopulistLib::Pop_RateAlgorithm;
using PopulistLib::PopulationConnection;
using PopulistLib::PopulationParameter;
using PopulistLib::PopulistSpecificParameter;
using PopulistLib::PopulistParameter;
using PopulistLib::Potential;
using DynamicLib::Rate;
using DynamicLib::Time;

int main(int argc, char* argv[])
{
	try {
	// In this example program, we reproduce the results presented in:
	//
	// A. Omurtag, Bruce W. Knight, L. Sirovich (2000),
	// On the Simulation of Large Populations of Neurons
	// Journal of Computational Neuroscience 8(1): 51-63 (2000)
	// 
	// This is the purest example of the use of the algorithm. More complicated, examples, involving networks of 
	// populations are described.


	Efficacy			H        = 0.03;  // synaptic effcacy of input synapse. The membrane potential is rescaled: 0 < v < 1
	Rate				nu       = 800;   // input rate Hz


	cout << "Most time is spent in the graphical rendering!" << endl;
	cout << "The algorithm is faster than this" << endl;
	cout << "h: "    << H  << endl;
	cout << "rate: " << nu << endl;

	// These are the parameters taken from that paper, don't blame us!

	Time      tau      = 50e-3; // a membrane time constant of 50 ms
	Time	  tau_ref  = 0;     //
	Potential theta	   = 1;     // (rescaled) neuron threshold
	Potential  V_reset = 0;     // reset potential
	Potential  V_rev   = 0;     // reversal potential

	// we need a network of two populations, one providing the input rate and one population 
	// of leaky-integrate-and-fire neurons
	Pop_Network network;    // really a DynamicNetwork<PopulistConnection>


	// create an input population
	Pop_RateAlgorithm  alg_input(nu); 
	NodeId id_input = network.AddNode(alg_input,EXCITATORY_BURST);

	// now create a population of leaky-integrated-and-fire neurons
	PopulationParameter 
		par_pop
		(
			theta,         
			V_reset,
			V_rev,
			tau_ref,
			tau
		);


	// Define some parameters, specific to the PopulistAlgorithm

	Potential V_min = -0.1;       // allow the grid to extend 10% below the reversal potential
	Number n_init_bins = 300;     // number of initial bins
	Number n_add       = 1;       // number of bins that is added after one zero-leak step

	InitialDensityParameter par_init(V_rev,0);  // a delta peak at the reversal potential

	double f_exp = 1.1;           // grid may expand by maximally 10%

	
	PopulistSpecificParameter 
		par_specific
		(
			V_min,
			n_init_bins,
			n_add,
			par_init,
			f_exp,
			"SingleInputZeroLeakEquations"
		);


	PopulationAlgorithm 
		alg_pop
		(
			PopulistParameter
			(
				par_pop,
				par_specific
			)
		);

	NodeId id_lif = network.AddNode(alg_pop,EXCITATORY_BURST);

	// now connect the two populations
	PopulationConnection con(1,H);                            
	network.MakeFirstInputOfSecond(id_input,id_lif,con);

	// visualisation and data logging object
	RootReportHandler 
		handler
		(
			"data.root",  // name of data file
			true,         // visualisation (try it switched off, it takes a lot of time)
			true          // write data into file
		);

	// decide the ranges to show in the canvas
	handler.SetFrequencyRange(0,20.);
	handler.SetDensityRange(-0.01,3);
	handler.SetTimeRange(0.,0.3);
	handler.SetPotentialRange(-0.1,1);

	handler.AddNodeToCanvas(id_lif);     // we want to visualize only the leaky-integrate-and-fire population

	SimulationRunParameter 
		par_run
		(
			handler,          // data saving object
			100000,           // maximum number of iterations (prevent infinite loops)
			0.,               // begin time of simulation
			0.3,              // end time of simulation
			1e-3,             // write out simulation results every 100 ms 
			1e-3,             // update visualisation every 100 ms
			1e-3,             // sample input every 10 ms
			"simulation.log"
		);

	// after a long set up we are ready to simulate:
	network.ConfigureSimulation(par_run);

	network.Evolve();
	
	// done
	}

	catch(UtilLib::GeneralException& excep)
	{
		cout << excep.Description() << endl;
	}

	return 0;
}
