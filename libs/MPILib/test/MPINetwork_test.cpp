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

#include <MPILib/include/algorithm/SleepAlgorithmCode.hpp>


#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib;

namespace mpi = boost::mpi;

mpi::communicator world;


void test_Constructor() {

	MPINetwork<double, utilities::CircularDistribution> network;

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._pNodeDistribution->isMaster()==true);
		BOOST_REQUIRE(network._pLocalNodes->size()==0);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._pLocalNodes->size()==0);
	}

	BOOST_REQUIRE(network._maxNodeId==0);
	BOOST_REQUIRE(network._current_report_time==0);
	BOOST_REQUIRE(network._current_state_time==0);
	BOOST_REQUIRE(network._current_simulation_time==0);
	BOOST_REQUIRE(network._isDalesLaw==true);
	//indirect test
	BOOST_REQUIRE(network._state_network._currentTime==0.0);
	BOOST_REQUIRE(network._parameter_simulation_run._t_begin==0);
	BOOST_REQUIRE(network._stream_log._b_time_available==true);
}

void test_AddNode() {

	MPINetwork<double, utilities::CircularDistribution> network;

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==0);
		BOOST_REQUIRE(network._pLocalNodes->size()==0);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._pLocalNodes->size()==0);
	}

	algorithm::SleepAlgorithm<double> alg;

	network.addNode(alg, EXCITATORY);

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==1);
		BOOST_REQUIRE(network._pLocalNodes->size()==1);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._pLocalNodes->size()==0);
	}

	network.addNode(alg, EXCITATORY);

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==2);
		BOOST_REQUIRE(network._pLocalNodes->size()==1);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._pLocalNodes->size()==1);
	}
}

void test_MakeFirstInputOfSecond() {

	MPINetwork<double, utilities::CircularDistribution> network;
	algorithm::SleepAlgorithm<double> alg;

	int node0 = network.addNode(alg, EXCITATORY);
	int node1 = network.addNode(alg, EXCITATORY);
	double weight = 2.0;

	bool exceptionThrown = false;
	try {
		network.makeFirstInputOfSecond(node0, node1, weight);
	} catch (...) {
		exceptionThrown = true;
	}
	BOOST_REQUIRE(exceptionThrown==false);
	if (world.rank() == 1) {

		BOOST_REQUIRE(
				network._pLocalNodes->find(node1)->second._precursors.size()==1);
		BOOST_REQUIRE(
				network._pLocalNodes->find(node1)->second._weights.size()==1);
		BOOST_REQUIRE(
				network._pLocalNodes->find(node1)->second._precursorActivity.size()==1);
	} else {
		BOOST_REQUIRE(
				network._pLocalNodes->find(node0)->second._successors.size()==1);
		BOOST_REQUIRE(
				network._pLocalNodes->find(node0)->second._weights.size()==0);
		BOOST_REQUIRE(
				network._pLocalNodes->find(node0)->second._precursorActivity.size()==0);
	}

	exceptionThrown = false;
	try {
		//use bad input to test exception
		network.makeFirstInputOfSecond(100, 101, weight);
	} catch (utilities::ParallelException &e) {
		exceptionThrown = true;
	}
	BOOST_REQUIRE(exceptionThrown==true);

}

void test_getMaxNodeId() {
	MPINetwork<double, utilities::CircularDistribution> network;
	algorithm::SleepAlgorithm<double> alg;
	BOOST_REQUIRE(network.getMaxNodeId()==0);
	network.addNode(alg, EXCITATORY);
	BOOST_REQUIRE(network.getMaxNodeId()==1);
	network.addNode(alg, EXCITATORY);
	network.addNode(alg, EXCITATORY);
	network.addNode(alg, EXCITATORY);
	BOOST_REQUIRE(network.getMaxNodeId()==4);

}

void test_incrementMaxNodeId() {
	MPINetwork<double, utilities::CircularDistribution> network;

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==0);
	}
	network.incrementMaxNodeId();
	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==1);
	}
	network.incrementMaxNodeId();
	network.incrementMaxNodeId();
	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==3);
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
//    BOOST_REQUIRE( add( 2,2 ) == 4 );      // #2 throws on error
//    if( add( 2,2 ) != 4 )
//        BOOST_ERROR( "Ouch..." );          // #3 continues on error
//    if( add( 2,2 ) != 4 )
//        BOOST_FAIL( "Ouch..." );           // #4 throws on error
//    if( add( 2,2 ) != 4 ) throw "Oops..."; // #5 throws on error
//
//    return add( 2, 2 ) == 4 ? 0 : 1;       // #6 returns error code
}
