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

/** @file
 *  This file replicates Omurtag A., Knight B. W., Sirovich L., On the simulation of large populations of neurons<i>J. of Comp. Neurosc.</i> (2000) ; 8(1)\
:51-63                     
 */ 
//! [preamble]
#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/algorithm/RateAlgorithmCode.hpp>
#include <MPILib/algorithm/RateFunctorCode.hpp>
#include <MPILib/algorithm/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/RootReportHandler.hpp>

using MPILib::EXCITATORY_DIRECT;
using MPILib::INHIBITORY_DIRECT;
using MPILib::NodeId;
using MPILib::SimulationRunParameter;
using MPILib::algorithm::RateAlgorithm;
using MPILib::algorithm::RateFunctor;
using MPILib::algorithm::WilsonCowanAlgorithm;
using MPILib::algorithm::WilsonCowanParameter;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<double, MPILib::utilities::CircularDistribution> Network;

Rate Inp(Time t){
  return (t > 1.0 && t < 1.3) ? 20.0 : 0;
}

Rate Opl(Time t){
  return (t > 0.5 && t < 1.5) ? 20.0 : 0;
}

int main(){
  
  Rate rate_ext = 0;

  RateFunctor<double> input(Inp);
  RateFunctor<double> openlock(Opl);

  RateAlgorithm<double> alg_ext(rate_ext);

	WilsonCowanParameter par_wc;
	par_wc._f_bias        = 0;
	par_wc._f_noise       = 1.0;
	par_wc._rate_maximum  = 50;
	par_wc._time_membrane = 50e-3;
  

	WilsonCowanAlgorithm alg(par_wc);

	Network network;

	NodeId id_input = network.addNode(input, EXCITATORY_DIRECT);
	NodeId id_gate = network.addNode(alg, EXCITATORY_DIRECT);
	NodeId id_output = network.addNode(alg, EXCITATORY_DIRECT);
	NodeId id_lock = network.addNode(alg, INHIBITORY_DIRECT);
	NodeId id_openlock = network.addNode(openlock ,INHIBITORY_DIRECT);

	double weight = 1.0;

	network.makeFirstInputOfSecond(id_input, id_gate, weight);
	network.makeFirstInputOfSecond(id_input, id_lock, weight);
	network.makeFirstInputOfSecond(id_gate, id_output, weight);
	network.makeFirstInputOfSecond(id_lock, id_gate, -1*weight);
	network.makeFirstInputOfSecond(id_openlock, id_lock, -2*weight);

	MPILib::CanvasParameter par_canvas;
	par_canvas._state_min     = -0.020;
	par_canvas._state_max     = 0.020;
	par_canvas._t_min         = 0.0;
	par_canvas._t_max         = 2.0;
	par_canvas._f_min         = 0.0;
	par_canvas._f_max         = 50.0;
	par_canvas._dense_min     = 0.0;
	par_canvas._dense_max     = 200.0;



	MPILib::report::handler::RootReportHandler handler("singlepoptest", true, true, par_canvas );

	handler.addNodeToCanvas(id_input);
	handler.addNodeToCanvas(id_gate);
	handler.addNodeToCanvas(id_output);
	handler.addNodeToCanvas(id_lock);
	handler.addNodeToCanvas(id_openlock);

	const SimulationRunParameter
		par_run
		(
		  handler,
		  10000000,
		  0.0,
		  2.0,
	          1e-3,
		  1e-3,
		   "wilsom.log"
		);

	network.configureSimulation(par_run);
	network.evolve();

	return 0;
}
