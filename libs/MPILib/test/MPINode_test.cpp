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
#include <MPILib/include/MPINodeCode.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#undef protected
#undef private
#include "helperClasses/SleepAlgorithm.hpp"

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	// make node global
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;
	NodeType nodeType = EXCITATORY;
	MPILib::NodeId nodeId = 1;
	MPINode<double, utilities::CircularDistribution> node(alg, nodeType, nodeId,
			network._pNodeDistribution, network._pLocalNodes);

// TODO DS test if the algorithm is the same
//	BOOST_CHECK(alg==node._algorithm);
	BOOST_CHECK(nodeType==node._nodeType);
	BOOST_CHECK(nodeId==node._nodeId);
	BOOST_CHECK(network._pNodeDistribution==node._pNodeDistribution);
	//indirect comparision
	BOOST_CHECK(network._pLocalNodes->size()==node._pLocalNodes->size());
	//make sure the shared_ptr works :)
	BOOST_CHECK(network._pNodeDistribution.use_count()==2);
}

void test_addPrecursor() {
	// make node global
	MPINetwork<double, utilities::CircularDistribution> network;

	SleepAlgorithm<double> alg;

	MPINode<double, utilities::CircularDistribution> node(alg, EXCITATORY, 1,
			network._pNodeDistribution, network._pLocalNodes);

	MPILib::NodeId nodeId = 4;
	double weight = 2.1;

	node.addPrecursor(nodeId, weight);
	BOOST_CHECK(node._precursors.size()==1);
	BOOST_CHECK(node._precursors[0]==4);
	BOOST_CHECK(node._weights.size()==1);
	BOOST_CHECK(node._weights[0]==2.1);
	BOOST_CHECK(node._precursorActivity.size()==1);

}

void test_addSuccessor() {
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;

	MPINode<double, utilities::CircularDistribution> node(alg, EXCITATORY, 1,
			network._pNodeDistribution, network._pLocalNodes);

	MPILib::NodeId nodeId = 4;

	node.addSuccessor(nodeId);
	BOOST_CHECK(node._precursors.size()==0);
	BOOST_CHECK(node._weights.size()==0);
	BOOST_CHECK(node._precursorActivity.size()==0);
	BOOST_CHECK(node._successors.size()==1);
	BOOST_CHECK(node._successors[0]==4);

}

void test_setGetState() {
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;

	MPINode<double, utilities::CircularDistribution> node(alg, EXCITATORY, 1,
			network._pNodeDistribution, network._pLocalNodes);

	node.setActivity(3);
	BOOST_CHECK(node.getActivity()==3);
	node.setActivity(4);
	BOOST_CHECK(node.getActivity()==4);
}

void test_sendRecvWait() {
	MPINode<double, utilities::CircularDistribution>* node;
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;
	if (world.rank() == 0) {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				EXCITATORY, 0, network._pNodeDistribution,
				network._pLocalNodes);

		node->addSuccessor(1);
		node->addPrecursor(1, 2.1);
	} else {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				EXCITATORY, 1, network._pNodeDistribution,
				network._pLocalNodes);

		node->addSuccessor(0);
		node->addPrecursor(0, 1.2);
	}

	node->setActivity(world.rank());
	node->sendOwnActivity();
	node->receiveData();
	node->waitAll();
	if (world.rank() == 0) {
		BOOST_CHECK(node->_precursorActivity[0]==1);
	} else {
		BOOST_CHECK(node->_precursorActivity[0]==0);
	}

	delete node;

}

void test_exchangeNodeTypes() {
	MPINode<double, utilities::CircularDistribution>* node;
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;
	if (world.rank() == 0) {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				EXCITATORY, 0, network._pNodeDistribution,
				network._pLocalNodes);

		node->addSuccessor(1);
		node->addPrecursor(1, 2.1);
	} else {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				INHIBITORY_BURST, 1, network._pNodeDistribution,
				network._pLocalNodes);

		node->addSuccessor(0);
		node->addPrecursor(0, 1.2);
	}

	node->exchangeNodeTypes();
	MPINode<double, utilities::CircularDistribution>::waitAll();
	if (world.rank() == 0) {
		BOOST_CHECK(node->_precursorTypes[0]==INHIBITORY_BURST);
	} else {
		BOOST_CHECK(node->_precursorTypes[0]==EXCITATORY);
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
	test_exchangeNodeTypes();

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
