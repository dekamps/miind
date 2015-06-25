// Copyright (c) 2005 - 2014 Marc de Kamps
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

#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/algorithm/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/RootReportHandler.hpp>

using GeomLib::CurrentCompensationParameter;
using GeomLib::DiffusionParameter;
using GeomLib::GeomAlgorithm;
using GeomLib::GeomParameter;
using GeomLib::InitialDensityParameter;
using GeomLib::OdeParameter;
using GeomLib::NeuronParameter;
using GeomLib::SpikingQifNeuralDynamics;
using GeomLib::QifOdeSystem;

using MPILib::EXCITATORY_DIRECT;
using MPILib::Efficacy;
using MPILib::NodeId;
using MPILib::SimulationRunParameter;
using MPILib::algorithm::RateAlgorithm;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;

int main(){

	cout << "Demonstrating Quadratic-Integrate-and-Fire under jump response" << endl;
	Number    n_bins = 1000;
	Potential V_min  = -10.0;

	NeuronParameter
		par_neuron
		(
			10.0,
			-10.0,
			-10.0,
			0.0,
			10e-3
		);

	OdeParameter
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
			InitialDensityParameter(0.0,0.0)
		);


	GeomLib::QifParameter
	  par_qif
	  (
	    0.5, // 
	    0.5  // default gamma sys
	  );

	DiffusionParameter par_diffusion(0.03,0.05);
	CurrentCompensationParameter par_current(0.0,0.0);
	SpikingQifNeuralDynamics dyn(par_ode,par_qif);
	QifOdeSystem sys(dyn);
	GeomDelayAlg alg_qif(GeomParameter(sys,  par_diffusion, par_current));
      


	// Slow decorrelation       
	Rate rate_ext = 5.0;
	Efficacy h    = 5.0;

	RateAlgorithm<MPILib::DelayedConnection> alg_ext(rate_ext);

	Network network;

	NodeId id_rate = network.addNode(alg_ext, MPILib::EXCITATORY_DIRECT);
	NodeId id_alg  = network.addNode(alg_qif, MPILib::EXCITATORY_DIRECT);

	MPILib::DelayedConnection con(1,h,0.0);
	network.makeFirstInputOfSecond(id_rate,id_alg,con);


	MPILib::CanvasParameter par_canvas;
	par_canvas._state_min     = -10.0;
	par_canvas._state_max     =  10.0;
	par_canvas._t_min         = 0.0;
	par_canvas._t_max         = 10.0;
	par_canvas._f_min         = 0.0;
	par_canvas._f_max         = 30.0;
	par_canvas._dense_min     = 0.0;
	par_canvas._dense_max     = 5.0;


	MPILib::report::handler::RootReportHandler handler("twopopcanvas.root", true, true, par_canvas);
	handler.addNodeToCanvas(id_alg);

      

	const SimulationRunParameter
		par_run
		(
			handler,
			10000000,
			0.0,
			1.0,
			2e-3,
			1e-4,
			"test/singlepoptest.log"
		);

	network.configureSimulation(par_run);
	network.evolve();

	return 0;
}
