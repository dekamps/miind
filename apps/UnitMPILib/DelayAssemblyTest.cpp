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

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/NodeType.hpp>
#include <MPILib/include/DelayAssemblyAlgorithmCode.hpp>
#include <MPILib/include/report/handler/RootReportHandler.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <GeomLib.hpp>

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;

using MPILib::RateFunctor;
using MPILib::DelayAssemblyParameter;
using MPILib::DelayAssemblyAlgorithm;
using MPILib::Time;




BOOST_AUTO_TEST_CASE(DelayAssemblyConstructionTest ) {
  
  MPILib::DelayAssemblyParameter par_ass;
  MPILib::DelayAssemblyAlgorithm<MPILib::DelayedConnection> alg_ass(par_ass);
}

Rate SwitchOnAndOff(Time t){
 
  if (t > 0.1 && t < 0.15)
    return 100.0;

  if (t > 0.7 && t < 0.75 )
    return -100.0;

  return 0.0;
}


BOOST_AUTO_TEST_CASE(DelayAssemblyNetworkTest) {
  Network network;

  DelayAssemblyParameter 
    par_ass
    (
     1e10, // A large membrane time constant; the population keeps its activation
     30.,  // If the population is switched on, it fires initially at 30 Hz
     10.,  // If the total weighted input comes above this value, delay is triggered
     -10.  // If the total weighted input comes below this value, delay is switched off
    );

  DelayAssemblyAlgorithm<MPILib::DelayedConnection> alg(par_ass);
  NodeId id_ass = network.addNode(alg,EXCITATORY_DIRECT);

  RateFunctor<MPILib::DelayedConnection> func(SwitchOnAndOff);
  NodeId id_switch = network.addNode(func,NEUTRAL);

  network.makeFirstInputOfSecond(id_switch,id_ass,MPILib::DelayedConnection(1.0,1.0,0.0));

  MPILib::report::handler::RootReportHandler handler("delayass.root", false, false);
  SimulationRunParameter par_run(handler,1000000,0.,1.,1e-2,1e-4,"delayass.log");
  network.configureSimulation(par_run);

  network.evolve();
}


