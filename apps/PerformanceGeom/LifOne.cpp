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
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/RootReportHandler.hpp>

using GeomLib::GeomAlgorithm;
using GeomLib::GeomParameter;
using GeomLib::InitialDensityParameter;
using GeomLib::LeakingOdeSystem;
using GeomLib::LifNeuralDynamics;
using GeomLib::OdeParameter;
using GeomLib::NeuronParameter;

using MPILib::EXCITATORY_DIRECT;
using MPILib::NodeId;
using MPILib::SimulationRunParameter;
using MPILib::RateAlgorithm;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;

int main(){

	cout << "Demonstrating Omurtag et al. (2000)" << endl;
	Number    n_bins = 500;
	Potential V_min  = 0.0;

	NeuronParameter
		par_neuron
		(
			1.0,
			0.0,
			0.0,
			0.0,
			50e-3
		);

	OdeParameter
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
			InitialDensityParameter(0.0,0.0)
		);

	double min_bin = 0.01;
	LifNeuralDynamics dyn(par_ode,min_bin);
	LeakingOdeSystem sys(dyn);
	GeomParameter par_geom(sys);
	GeomDelayAlg alg(par_geom);

	Rate rate_ext = 800.0;
	RateAlgorithm<MPILib::DelayedConnection> alg_ext(rate_ext);

	Network network;

	NodeId id_rate = network.addNode(alg_ext,EXCITATORY_DIRECT);
	NodeId id_alg  = network.addNode(alg,    EXCITATORY_DIRECT);

	MPILib::DelayedConnection con(1,0.03,0.0);
	network.makeFirstInputOfSecond(id_rate,id_alg,con);

	const MPILib::report::handler::RootReportHandler handler("singlepoptest", true, false);

	const SimulationRunParameter
		par_run
		(
			handler,
			10000000,
			0.0,
			10.0,
			1e-4,
			1e-4,
			"singlepoptest.log"
		);

	network.configureSimulation(par_run);
	network.evolve();

	return 0;
}
