/*
 * MPINetwork_test.cpp
 *
 *  Created on: 31.05.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

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
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

#include "helperClasses/SleepAlgorithm.hpp"


namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	MPINetwork<double, utilities::CircularDistribution> network;

	if (world.rank() == 0) {
		BOOST_CHECK(network._pNodeDistribution->isMaster()==true);
		BOOST_CHECK(network._pLocalNodes->size()==0);
	} else if (world.rank() == 1) {
		BOOST_CHECK(network._pLocalNodes->size()==0);
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

	if (world.rank() == 0) {
		BOOST_CHECK(network._maxNodeId==0);
		BOOST_CHECK(network._pLocalNodes->size()==0);
	} else if (world.rank() == 1) {
		BOOST_CHECK(network._pLocalNodes->size()==0);
	}

	SleepAlgorithm<double> alg;

	network.addNode(alg, EXCITATORY);

	if (world.rank() == 0) {
		BOOST_CHECK(network._maxNodeId==1);
		BOOST_CHECK(network._pLocalNodes->size()==1);
	} else if (world.rank() == 1) {
		BOOST_CHECK(network._pLocalNodes->size()==0);
	}

	network.addNode(alg, EXCITATORY);

	if (world.rank() == 0) {
		BOOST_CHECK(network._maxNodeId==2);
		BOOST_CHECK(network._pLocalNodes->size()==1);
	} else if (world.rank() == 1) {
		BOOST_CHECK(network._pLocalNodes->size()==1);
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
	if (world.rank() == 1) {

		BOOST_CHECK(
				network._pLocalNodes->find(node1)->second._precursors.size()==1);
		BOOST_CHECK(
				network._pLocalNodes->find(node1)->second._weights.size()==1);
		BOOST_CHECK(
				network._pLocalNodes->find(node1)->second._precursorActivity.size()==1);
	} else {
		BOOST_CHECK(
				network._pLocalNodes->find(node0)->second._successors.size()==1);
		BOOST_CHECK(
				network._pLocalNodes->find(node0)->second._weights.size()==0);
		BOOST_CHECK(
				network._pLocalNodes->find(node0)->second._precursorActivity.size()==0);
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

	if (world.rank() == 0) {
		BOOST_CHECK(network._maxNodeId==0);
	}
	network.incrementMaxNodeId();
	if (world.rank() == 0) {
		BOOST_CHECK(network._maxNodeId==1);
	}
	network.incrementMaxNodeId();
	network.incrementMaxNodeId();
	if (world.rank() == 0) {
		BOOST_CHECK(network._maxNodeId==3);
	}
}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

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
