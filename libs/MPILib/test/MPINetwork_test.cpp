// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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
#include <MPILib/config.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
#endif
#include <MPILib/include/utilities/MPIProxy.hpp>
MPILib::utilities::MPIProxy mpiProxy;
//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/MPINetworkCode.hpp>
#undef protected
#undef private

#include <MPILib/include/NodeType.hpp>

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib;

#include <boost/thread/thread.hpp>

#include "helperClasses/SleepAlgorithm.hpp"

void test_Constructor() {

	MPINetwork<double, utilities::CircularDistribution> network;

	if (mpiProxy.getRank() == 0) {
		BOOST_CHECK(network._nodeDistribution.isMaster()==true);
		BOOST_CHECK(network._localNodes.size()==0);
	} else if (mpiProxy.getRank() == 1) {
		BOOST_CHECK(network._localNodes.size()==0);
	}

	BOOST_CHECK(network._maxNodeId==0);
	BOOST_CHECK(network._currentReportTime==0);
	BOOST_CHECK(network._currentStateTime==0);
	BOOST_CHECK(network._currentSimulationTime==0);
	BOOST_CHECK(network._isDalesLaw==true);
	//indirect test
	BOOST_CHECK(network._stateNetwork._currentTime==0.0);
	BOOST_CHECK(network._parameterSimulationRun._tBegin==0);
	BOOST_CHECK(network._streamLog._isTimeAvailable==true);
}

void test_AddNode() {

	MPINetwork<double, utilities::CircularDistribution> network;

	if (mpiProxy.getRank() == 0) {
		BOOST_CHECK(network._maxNodeId==0);
		BOOST_CHECK(network._localNodes.size()==0);
	} else if (mpiProxy.getRank() == 1) {
		BOOST_CHECK(network._localNodes.size()==0);
	}

	SleepAlgorithm<double> alg;

	network.addNode(alg, EXCITATORY);

	if (mpiProxy.getRank() == 0) {
		BOOST_CHECK(network._maxNodeId==1);
		BOOST_CHECK(network._localNodes.size()==1);
	} else if (mpiProxy.getRank() == 1) {
		BOOST_CHECK(network._localNodes.size()==0);
	}

	network.addNode(alg, EXCITATORY);

	if (mpiProxy.getRank() == 0) {
		BOOST_CHECK(network._maxNodeId==2);
		if (mpiProxy.getSize() == 1) {
			BOOST_CHECK(network._localNodes.size()==2);
		} else {
			BOOST_CHECK(network._localNodes.size()==1);
		}
	} else if (mpiProxy.getRank() == 1) {
		BOOST_CHECK(network._localNodes.size()==1);
	}
}

void test_MakeFirstInputOfSecond() {

	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;

	int node0 = network.addNode(alg, EXCITATORY);
	int node1 = network.addNode(alg, EXCITATORY);
	double weight = 2.0;

	bool exceptionThrown = false;
	try {
		network.makeFirstInputOfSecond(node0, node1, weight);
	} catch (...) {
		exceptionThrown = true;
	}
	BOOST_CHECK(exceptionThrown==false);
	if (mpiProxy.getRank() == 1) {

		BOOST_CHECK(
				network._localNodes.find(node1)->second._precursors.size()==1);
		BOOST_CHECK(
				network._localNodes.find(node1)->second._weights.size()==1);
		BOOST_CHECK(
				network._localNodes.find(node1)->second._precursorActivity.size()==1);
	} else {
		BOOST_CHECK(
				network._localNodes.find(node0)->second._successors.size()==1);
		BOOST_CHECK(
				network._localNodes.find(node0)->second._weights.size()==0);
		BOOST_CHECK(
				network._localNodes.find(node0)->second._precursorActivity.size()==0);
	}

	exceptionThrown = false;
	try {
		//use bad input to test exception
		network.makeFirstInputOfSecond(100, 101, weight);
	} catch (utilities::ParallelException &e) {
		exceptionThrown = true;
	}
	BOOST_CHECK(exceptionThrown==true);

}

void test_getMaxNodeId() {
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;
	BOOST_CHECK(network.getMaxNodeId()==0);
	network.addNode(alg, EXCITATORY);
	BOOST_CHECK(network.getMaxNodeId()==1);
	network.addNode(alg, EXCITATORY);
	network.addNode(alg, EXCITATORY);
	network.addNode(alg, EXCITATORY);
	BOOST_CHECK(network.getMaxNodeId()==4);

}

void test_incrementMaxNodeId() {
	MPINetwork<double, utilities::CircularDistribution> network;

	if (mpiProxy.getRank() == 0) {
		BOOST_CHECK(network._maxNodeId==0);
	}
	network.incrementMaxNodeId();
	if (mpiProxy.getRank() == 0) {
		BOOST_CHECK(network._maxNodeId==1);
	}
	network.incrementMaxNodeId();
	network.incrementMaxNodeId();
	if (mpiProxy.getRank() == 0) {
		BOOST_CHECK(network._maxNodeId==3);
	}
}

int test_main(int argc, char* argv[]) // note the name!
		{

#ifdef ENABLE_MPI
	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing
	if (mpiProxy.getSize() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
#endif

	test_Constructor();
	test_AddNode();
	test_MakeFirstInputOfSecond();
	test_getMaxNodeId();
	test_incrementMaxNodeId();

	return 0;
//    // six ways to detect and report the same error:
//    BOOST_CHECK( add( 2,2 ) == 4 );        // #1 continues on error
//    BOOST_CHECK( add( 2,2 ) == 4 );      // #2 throws on error
//    if( add( 2,2 ) != 4 )
//        BOOST_ERROR( "Ouch..." );          // #3 continues on error
//    if( add( 2,2 ) != 4 )
//        BOOST_FAIL( "Ouch..." );           // #4 throws on error
//    if( add( 2,2 ) != 4 ) throw "Oops..."; // #5 throws on error
//
//    return add( 2, 2 ) == 4 ? 0 : 1;       // #6 returns error code
}
