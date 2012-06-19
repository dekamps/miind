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
#include <MPILib/include/algorithm/SleepAlgorithmCode.hpp>

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
	algorithm::SleepAlgorithm<double> alg;
	NodeType nodeType = EXCITATORY;
	MPILib::NodeId nodeId = 1;
	MPINode<double, utilities::CircularDistribution> node(alg, nodeType, nodeId,
			network._pNodeDistribution, network._pLocalNodes);

// TODO test if the algorithm is the same
//	BOOST_REQUIRE(alg==node._algorithm);
	BOOST_REQUIRE(nodeType==node._nodeType);
	BOOST_REQUIRE(nodeId==node._nodeId);
	BOOST_REQUIRE(network._pNodeDistribution==node._pNodeDistribution);
	//indirect comparision
	BOOST_REQUIRE(network._pLocalNodes->size()==node._pLocalNodes->size());
	//make sure the shared_ptr works :)
	BOOST_REQUIRE(network._pNodeDistribution.use_count()==2);
}

void test_addPrecursor() {
	// make node global
	MPINetwork<double, utilities::CircularDistribution> network;

	algorithm::SleepAlgorithm<double> alg;

	MPINode<double, utilities::CircularDistribution> node(alg, EXCITATORY,1,
			network._pNodeDistribution, network._pLocalNodes);

	MPILib::NodeId nodeId = 4;
	double weight = 2.1;

	node.addPrecursor(nodeId, weight);
	BOOST_REQUIRE(node._precursors.size()==1);
	BOOST_REQUIRE(node._precursors[0]==4);
	BOOST_REQUIRE(node._weights.size()==1);
	BOOST_REQUIRE(node._weights[0]==2.1);
	BOOST_REQUIRE(node._precursorActivity.size()==1);

}

void test_addSuccessor() {
	MPINetwork<double, utilities::CircularDistribution> network;
	algorithm::SleepAlgorithm<double> alg;

	MPINode<double, utilities::CircularDistribution> node(alg, EXCITATORY,1,
			network._pNodeDistribution, network._pLocalNodes);

	MPILib::NodeId nodeId = 4;

	node.addSuccessor(nodeId);
	BOOST_REQUIRE(node._precursors.size()==0);
	BOOST_REQUIRE(node._weights.size()==0);
	BOOST_REQUIRE(node._precursorActivity.size()==0);
	BOOST_REQUIRE(node._successors.size()==1);
	BOOST_REQUIRE(node._successors[0]==4);

}

void test_setGetState() {
	MPINetwork<double, utilities::CircularDistribution> network;
	algorithm::SleepAlgorithm<double> alg;

	MPINode<double, utilities::CircularDistribution> node(alg, EXCITATORY, 1,
			network._pNodeDistribution, network._pLocalNodes);

	node.setActivity(3);
	BOOST_REQUIRE(node.getActivity()==3);
	node.setActivity(4);
	BOOST_REQUIRE(node.getActivity()==4);
}

void test_sendRecvWait() {
	MPINode<double, utilities::CircularDistribution>* node;
	MPINetwork<double, utilities::CircularDistribution> network;
	algorithm::SleepAlgorithm<double> alg;
	if (world.rank() == 0) {

		node = new MPINode<double, utilities::CircularDistribution>(alg, EXCITATORY, 0,
				network._pNodeDistribution, network._pLocalNodes);

		node->addSuccessor(1);
		node->addPrecursor(1, 2.1);
	} else {

		node = new MPINode<double, utilities::CircularDistribution>(alg, EXCITATORY, 1,
				network._pNodeDistribution, network._pLocalNodes);

		node->addSuccessor(0);
		node->addPrecursor(0, 1.2);
	}

	node->setActivity(world.rank());
	node->sendOwnActivity();
	node->receiveData();
	node->waitAll();
	if (world.rank() == 0) {
		BOOST_REQUIRE(node->_precursorActivity[0]==1);
	} else {
		BOOST_REQUIRE(node->_precursorActivity[0]==0);
	}

	delete node;

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
