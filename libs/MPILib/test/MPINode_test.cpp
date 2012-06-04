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
#include <MPILib/include/MPINode.hpp>
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

	// make node global
	MPINetwork network;
	Algorithm alg = 1;
	NodeType nodeType = 1;
	NodeId nodeId = 1;
	MPINode node(alg, nodeType, nodeId, network._nodeDistribution,
			network._localNodes);

	BOOST_REQUIRE(alg==node._algorithm);
	BOOST_REQUIRE(nodeType==node._nodeType);
	BOOST_REQUIRE(nodeId==node._nodeId);
	BOOST_REQUIRE(network._nodeDistribution==node._nodeDistribution);
	//indirect comparision
	BOOST_REQUIRE(network._localNodes.size()==node._refLocalNodes.size());
	//make sure the shared_ptr works :)
	BOOST_REQUIRE(network._nodeDistribution.use_count()==2);
}

void test_addPrecursor() {
	// make node global
	MPINetwork network;
	MPINode node(1, 1, 1, network._nodeDistribution, network._localNodes);

	NodeId nodeId = 4;
	WeightType weight = 2.1;

	node.addPrecursor(nodeId, weight);
	BOOST_REQUIRE(node._precursors.size()==1);
	BOOST_REQUIRE(node._precursors[0]==4);
	BOOST_REQUIRE(node._weights.size()==1);
	BOOST_REQUIRE(node._weights[0]==2.1);
	BOOST_REQUIRE(node._precursorStates.size()==1);

}

void test_addSuccessor() {
	MPINetwork network;
	MPINode node(1, 1, 1, network._nodeDistribution, network._localNodes);

	NodeId nodeId = 4;

	node.addSuccessor(nodeId);
	BOOST_REQUIRE(node._precursors.size()==0);
	BOOST_REQUIRE(node._weights.size()==0);
	BOOST_REQUIRE(node._precursorStates.size()==0);
	BOOST_REQUIRE(node._successors.size()==1);
	BOOST_REQUIRE(node._successors[0]==4);

}

void test_setGetState() {
	MPINetwork network;
	MPINode node(1, 1, 1, network._nodeDistribution, network._localNodes);
	node.setState(3);
	BOOST_REQUIRE(node.getState()==3);
	node.setState(4);
	BOOST_REQUIRE(node.getState()==4);
}

void test_sendRecvWait() {
	MPINode* node;
	MPINetwork network;
	if (world.rank() == 0) {
		node = new MPINode(1, 1, 0, network._nodeDistribution,
				network._localNodes);
		node->addSuccessor(1);
		node->addPrecursor(1, 2.1);
	} else {
		node = new MPINode(1, 1, 1, network._nodeDistribution,
				network._localNodes);
		node->addSuccessor(0);
		node->addPrecursor(0, 1.2);
	}

	node->setState(world.rank());
	node->sendOwnState();
	node->receiveData();
	node->waitAll();
	if (world.rank() == 0) {
		BOOST_REQUIRE(node->_precursorStates[0]==1);
	} else {
		BOOST_REQUIRE(node->_precursorStates[0]==0);
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
	test_addPrecursor();
	test_addSuccessor();
	test_setGetState();
	test_sendRecvWait();

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
