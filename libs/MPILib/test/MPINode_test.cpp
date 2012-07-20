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

void test_Constructor() {

	// make node global
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;
	NodeType nodeType = EXCITATORY;
	MPILib::NodeId nodeId = 1;
	MPINode<double, utilities::CircularDistribution> node(alg, nodeType, nodeId,
			network._nodeDistribution, network._localNodes);

// TODO DS test if the algorithm is the same
//	BOOST_CHECK(alg==node._algorithm);
	BOOST_CHECK(nodeType==node._nodeType);
	BOOST_CHECK(nodeId==node._nodeId);
	//indirect test
	BOOST_CHECK(
			network._nodeDistribution.isMaster()==node._rNodeDistribution.isMaster());
	//indirect comparision
	BOOST_CHECK(network._localNodes.size()==node._rLocalNodes.size());
}

void test_addPrecursor() {
	// make node global
	MPINetwork<double, utilities::CircularDistribution> network;

	SleepAlgorithm<double> alg;

	MPINode<double, utilities::CircularDistribution> node(alg, EXCITATORY, 1,
			network._nodeDistribution, network._localNodes);

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
			network._nodeDistribution, network._localNodes);

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
			network._nodeDistribution, network._localNodes);

	node.setActivity(3);
	BOOST_CHECK(node.getActivity()==3);
	node.setActivity(4);
	BOOST_CHECK(node.getActivity()==4);
}

void test_sendRecvWait() {
	MPINode<double, utilities::CircularDistribution>* node;
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;
	MPILib::utilities::MPIProxy mpiProxy;

	if (mpiProxy.getRank() == 0) {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				EXCITATORY, 0, network._nodeDistribution, network._localNodes);

		node->addSuccessor(1);
		node->addPrecursor(1, 2.1);
	} else {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				EXCITATORY, 1, network._nodeDistribution, network._localNodes);

		node->addSuccessor(0);
		node->addPrecursor(0, 1.2);
	}

	node->setActivity(mpiProxy.getRank());
	node->sendOwnActivity();
	node->receiveData();
	MPINode<double, utilities::CircularDistribution>::waitAll();
	if (mpiProxy.getSize() == 2) {
		if (mpiProxy.getRank() == 0) {
			BOOST_CHECK(node->_precursorActivity[0]==1);
		} else {
			BOOST_CHECK(node->_precursorActivity[0]==0);
		}
	} else {
		BOOST_CHECK(node->_precursorActivity[1]==0);

	}

	delete node;

}

void test_exchangeNodeTypes() {
	MPINode<double, utilities::CircularDistribution>* node;
	MPINetwork<double, utilities::CircularDistribution> network;
	SleepAlgorithm<double> alg;
	MPILib::utilities::MPIProxy mpiProxy;

	if (mpiProxy.getRank() == 0) {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				EXCITATORY, 0, network._nodeDistribution, network._localNodes);

		node->addSuccessor(1);
		node->addPrecursor(1, 2.1);
	} else {

		node = new MPINode<double, utilities::CircularDistribution>(alg,
				INHIBITORY_BURST, 1, network._nodeDistribution,
				network._localNodes);

		node->addSuccessor(0);
		node->addPrecursor(0, 1.2);
	}

	node->exchangeNodeTypes();
	MPINode<double, utilities::CircularDistribution>::waitAll();
	if (mpiProxy.getSize() == 2) {
		if (mpiProxy.getRank() == 0) {
			BOOST_CHECK(node->_precursorTypes[0]==INHIBITORY_BURST);
		} else {
			BOOST_CHECK(node->_precursorTypes[0]==EXCITATORY);
		}
	}
	delete node;

}

int test_main(int argc, char* argv[]) // note the name!
		{

#ifdef ENABLE_MPI
	boost::mpi::environment env(argc, argv);
	MPILib::utilities::MPIProxy mpiProxy;

	// we use only two processors for this testing
	if (mpiProxy.getSize() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
#endif

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
