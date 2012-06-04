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
#include <MPILib/include/MPINetwork.hpp>
#undef protected
#undef private

#include <MPILib/include/utilities/ParallelException.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	MPINetwork network;

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._nodeDistribution->isMaster()==true);
		BOOST_REQUIRE(network._localNodes.size()==0);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._localNodes.size()==0);
	}

}

void test_AddNode() {

	MPINetwork network;

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==0);
		BOOST_REQUIRE(network._localNodes.size()==0);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._localNodes.size()==0);
	}

	network.AddNode(1, 1);

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==1);
		BOOST_REQUIRE(network._localNodes.size()==1);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._localNodes.size()==0);
	}

	network.AddNode(1, 1);

	if (world.rank() == 0) {
		BOOST_REQUIRE(network._maxNodeId==2);
		BOOST_REQUIRE(network._localNodes.size()==1);
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(network._localNodes.size()==1);
	}
}

void test_MakeFirstInputOfSecond() {

	MPINetwork network;

	int node0 = network.AddNode(1, 1);
	int node1 = network.AddNode(1, 1);
	double weight = 2.0;

	bool exceptionThrown = false;
	try {
		network.MakeFirstInputOfSecond(node0, node1, weight);
	} catch (...) {
		exceptionThrown = true;
	}
	BOOST_REQUIRE(exceptionThrown==false);
	if (world.rank() == 1) {

		BOOST_REQUIRE(
				network._localNodes.find(node1)->second._precursors.size()==1);
		BOOST_REQUIRE(
				network._localNodes.find(node1)->second._weights.size()==1);
		BOOST_REQUIRE(
				network._localNodes.find(node1)->second._precursorStates.size()==1);
	} else {
		BOOST_REQUIRE(
				network._localNodes.find(node0)->second._successors.size()==1);
		BOOST_REQUIRE(
				network._localNodes.find(node0)->second._weights.size()==0);
		BOOST_REQUIRE(
				network._localNodes.find(node0)->second._precursorStates.size()==0);
	}

	exceptionThrown = false;
	try {
		//use bad input to test exception
		network.MakeFirstInputOfSecond(100, 101, weight);
	} catch (utilities::ParallelException &e) {
		exceptionThrown = true;
	}
	BOOST_REQUIRE(exceptionThrown==true);

}

void test_getMaxNodeId() {
	MPINetwork network;

	BOOST_REQUIRE(network.getMaxNodeId()==0);
	network.AddNode(1, 1);
	BOOST_REQUIRE(network.getMaxNodeId()==1);
	network.AddNode(1, 1);
	network.AddNode(1, 1);
	network.AddNode(1, 1);
	BOOST_REQUIRE(network.getMaxNodeId()==4);

}

void test_incrementMaxNodeId() {
	MPINetwork network;

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
