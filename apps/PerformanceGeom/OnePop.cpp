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
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/MinimalReportHandler.hpp>


using MPILib::EXCITATORY_DIRECT;
using MPILib::NodeId;
using MPILib::SimulationRunParameter;
using MPILib::RateFunctor;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;

Rate RATE;

MPILib::Rate Func(MPILib::Time t){
  return (t < 0.5) ? RATE : 0.0;
    }

int main(int argc, char* argv[]){

	if (argc != 5){
		std::cout << "Use: ./Onepop <modelname> <matname> rate efficacy" << std::endl;
		exit(0);
	}
	try {
		// Processing command line
		const string mesh_name = argv[1];
		vector<string> mat_names;
		mat_names.push_back(argv[2]);

		Time h = 1e-4;

		TwoDLib::MeshAlgorithm<MPILib::DelayedConnection> algmesh(mesh_name,mat_names,h);

		std::istringstream ist_rate(argv[3]);
		Rate rate_ext;
		ist_rate >> rate_ext;
		RATE = rate_ext;

		RateFunctor<MPILib::DelayedConnection> alg_ext(Func);

		std::istringstream ist_eff(argv[4]);
		Efficacy eff;
		ist_eff >> eff;

		// Network creation
		Network network;

		NodeId id_rate = network.addNode(alg_ext,NEUTRAL);
		NodeId id_alg  = network.addNode(algmesh,EXCITATORY_DIRECT);

		MPILib::DelayedConnection con(1,eff,0.0);

		network.makeFirstInputOfSecond(id_rate,id_alg,con);

		const MPILib::report::handler::MinimalReportHandler handler("onepop_result");

		const SimulationRunParameter
		par_run
		(
			handler,
			10000000,
			0.0,
			1.0,
			1e-3,
			algmesh.MeshReference().TimeStep(),
			"singlepoptest.log"
		);
		network.configureSimulation(par_run);
		network.evolve();
	}
	catch(TwoDLib::TwoDLibException& excep){
		std::cout << excep.Description() << std::endl;
	}
	catch(...){
		std::cout << "Something happened" << std::endl;
	}
	return 0;
}
