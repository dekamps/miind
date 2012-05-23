// Copyright (c) 2005 - 2011 Marc de Kamps
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

#include <fstream>
#include <algorithm>
#include "../NetLib/NetLib.h"
#include "LayerWeightIteratorThresholdCode.h"
#include "SparseImplementationTestCode.h"
#include "ReversibleSparseNode.h"
#include "ReversibleLayeredSparseImplementationCode.h"
#include "SparseImplementationAllocator.h"
#include "SparseNodeCreationExample.h"
#include "SparseLibIteratorException.h"
#include "SparseLibTestOpenException.h"



using namespace std;
using namespace NetLib;
using namespace SparseImplementationLib;

SparseImplementationTest::SparseImplementationTest()
{
}

SparseImplementationTest::SparseImplementationTest
(
	boost::shared_ptr<ostream> p_stream
):
LogStream(p_stream)
{
}

bool SparseImplementationTest::Execute()
{
	NetLibTest test(Stream());
	if (! test.Execute())
		return false;
	Record("NetLibTest succeeded");

	if (! SparseNodeCreationTest() )
		return false;
	Record("SparseNodeCreationTest succeeded");

	if (! PredecessorIteratorTest() )
		return false;
	Record("PredecessorIteratorTest succeeded");

	if (! SparseNodeCopyTest() )
		return false;
	Record("SparseNodeCopyTest succeeded");

	if (! NodeVectorTest() )
		return false;
	Record("NodeVectorTest succeeded");

	if (! SparseImplementationCopyTest() )
		return false;
	Record("SparseImplementationCopyTest");

	if (! SparseImplementationTesting() )
		return false;
	Record("SparseImplementationTesting succeeded");

	if (! SparseImplementationReversalTest() )
		return false;
	Record("SparseImplementationReversalTest succeeded");

	if (! LayerOrderTest() )
		return false;
	Record("LayerOrdertest");

	if (! LayerIteratorTest() )
		return false;
	Record("LayerIteratorTest");

	if (! LayerIteratorThresholdTest() )
		return false;
	Record("LayerIteratorThresholdTest succeeded");

	if (! ReverseIteratorTest() )
		return false;
	Record("ReverseIteratorTest succeeded");

	if (! ReverseIteratorThresholdTest() )
		return false;
	Record("ReverseIteratorThresholdTest succeeded");

	if (! XORImplementationTest() )
		return false;
	Record("XORImplementationTest succeeded");

	if (! TestSparseNodeStreaming() )
		return false;
	Record("TestSparseNodeStreaming");

	if (! TestImplementationStreaming() )
		return false;
	Record("TestImplementationStreaming");

	if (! DynamicInsertionTest() )
		return false;
	Record("DynamicInsertionTest succeeded");
	
	if (! SparseImplementationAllocatorTest() )
		return false;
	Record("SparseImplementationAllocatorTest");
	
	if (! SparseNodeCreationExample() )
		return false;
	Record("SparseNodeCreationExampleTest succeeded");

	if (! ImplementationCreationTest() )
		return false;
	Record("ImplementationCreationTest succeeded");

	if (! NodeLinkCollectionStreamTest() )
		return false;
	Record("NodeLinkCollectionStreamTest succeeded");

	if (! NavigationTest() )
		return false;
	Record("NavigationTest succeeded");

	return true;
}

bool SparseImplementationTest::SparseNodeCreationTest() const
{
        return SparseNodeCreationExample();
}

bool SparseImplementationTest::SparseNodeCopyTest() const
{	
	D_SparseNode node_1;
	node_1.SetId(NodeId(1));

	D_SparseNode node_2;

	// assingment operator
	node_2 = node_1;

	// copy constructor
	D_SparseNode node_3 = node_1;

	return true;
}

bool SparseImplementationTest::SparseImplementationCopyTest() const
{
	// Ordinary SparseNodes
	SparseImplementationTest test;
	D_LayeredSparseImplementation implementation = test.CreateTestLayeredSparseImplementation<D_SparseNode>();

	// create a one layered architecture
	vector<Layer> layer_vector;
	layer_vector.push_back(10);
	LayeredArchitecture  arch(layer_vector);
	D_LayeredSparseImplementation implementation_copy_operated(&arch);
	implementation_copy_operated = implementation;

	D_LayeredSparseImplementation implementation_copy_constructed = implementation;

	// ReversibleSparseNode
	D_LayeredReversibleSparseImplementation 
		implementation_reversible = test.CreateTestReversibleLayeredSparseImplementation<D_ReversibleSparseNode>();


	// Can't take the old Architecture !!
	vector<Layer> vector_layer;
	vector_layer.push_back(10);
	LayeredArchitecture arch_2(vector_layer);
	D_LayeredReversibleSparseImplementation implementation_reversible_copy_operated(&arch_2);
	implementation_reversible_copy_operated = implementation_reversible;

	D_LayeredReversibleSparseImplementation implementation_reversible_copy_constructed = implementation_reversible;

	return true;
}

bool SparseImplementationTest::SparseImplementationReversalTest() const
{
	SparseImplementationTest test;

	D_LayeredReversibleSparseImplementation 
		implementation = test.CreateTestReversibleLayeredSparseImplementation<D_ReversibleSparseNode>();

	implementation.InsertReverseImplementation();

	return implementation.IsReverseImplementationConsistent();
}

bool SparseImplementationTest::SparseImplementationTesting() const
{
	SparseImplementationTest test;

	D_LayeredSparseImplementation implementation 
		= test.CreateTestLayeredSparseImplementation<D_SparseNode>();

	return true;
}

bool SparseImplementationTest::XORImplementationTest() const
{
	LayeredSparseImplementation<D_ReversibleSparseNode> implementation =
		CreateXORImplementation();

	return true;
}

LayeredArchitecture 
	SparseImplementationTest::CreateXORArchitecture() const
{
 	// Create link vector
	vector<NodeLink> vector_link;

	// Create input layer
	NodeId One(1);
	NodeId Two(2);
	vector<NodeId> vector_empty(0);

	NodeLink link_one(One,vector_empty);
	NodeLink link_two(Two,vector_empty);

	vector_link.push_back(link_one);
	vector_link.push_back(link_two);

	// Middle layer
	NodeId Three(3);
	NodeId Four(4);

	vector<NodeId> vector_input_connection;
	vector_input_connection.push_back(One);
	vector_input_connection.push_back(Two);

	vector_link.push_back(NodeLink(Three,vector_input_connection));
	vector_link.push_back(NodeLink(Four, vector_input_connection));

	// Output Layer
	NodeId Five(5);
	vector<NodeId> vector_middle_connection;
	vector_middle_connection.push_back(Three);
	vector_middle_connection.push_back(Four);

	vector_link.push_back(NodeLink(Five,vector_middle_connection));

	NodeLinkCollection* p_collection = new NodeLinkCollection(vector_link);

	// Set up architecture

	vector<Layer> vector_architecture(3);
	vector_architecture[0] = 2;
	vector_architecture[1] = 2;
	vector_architecture[2] = 1;

	LayeredArchitecture architecture = 
		LayeredArchitecture
		(
			vector_architecture,
			p_collection,
			true                    // essential for XOR !
		);

	return architecture;
}

LayeredSparseImplementation<D_ReversibleSparseNode> 
	SparseImplementationTest::CreateXORImplementation() const
{
	LayeredArchitecture architecture = CreateXORArchitecture();

	// and implementation
	D_LayeredReversibleSparseImplementation implementation(&architecture);


	return implementation;
}


bool SparseImplementationTest::LayerIteratorTest() const
{
	D_LayeredSparseImplementation implementation = CreateTestLayeredSparseImplementation<D_SparseNode>();

	D_LayeredSparseImplementation::WeightLayerIterator iter;

	int index = 1; // First NodeId for comparison
	for (Layer n_layer = 0; n_layer < implementation.NumberOfLayers(); n_layer++)
	{
		typedef D_LayeredSparseImplementation::WeightLayerIterator iterator;
		iterator* p_dummy = 0;
		iterator iter_begin = implementation.begin(n_layer,p_dummy);
        iterator iter_end   = implementation.end  (n_layer,p_dummy);

		// This should iterate over all 10 Nodes in the implementation
		for (iter = iter_begin; iter != iter_end; iter++)
			if (index++ != iter->MyNodeId()._id_value)
				return false;
	}
	return true;
}


bool SparseImplementationTest::LayerIteratorThresholdTest() const
{
	D_LayeredSparseImplementation implementation = CreateTestLayeredSparseImplementation<D_SparseNode>();

	D_LayeredSparseImplementation::WeightLayerIteratorThreshold iter;

	int index = 1; // First NodeId for comparisom
	for (Layer n_layer = 0; n_layer < implementation.NumberOfLayers(); n_layer++)
	{
		typedef D_LayeredSparseImplementation::WeightLayerIteratorThreshold iterator; 
		iterator* p_dummy = 0;
		iterator iter_begin = implementation.begin(n_layer, p_dummy);
		iterator iter_end   = implementation.end  (n_layer, p_dummy);

		// The first Node should NodeId(0)
		iter = iter_begin;
		if (iter->MyNodeId() != THRESHOLD_ID)
			return false;

		// This should iterate over all 10 Nodes in the implementation
		for (iter = ++iter_begin; iter != iter_end; iter++)
			if (index++ != iter->MyNodeId()._id_value)
				return false;
	}
	return true;
}


bool SparseImplementationTest::TestReverseInnerProductLayer1
	(
		D_LayeredReversibleSparseImplementation& implementation
	) const
{
	typedef D_LayeredReversibleSparseImplementation::WeightLayerIterator iterator;
	iterator* p_dummy = 0;
	iterator iter = implementation.begin(0,p_dummy);

	implementation.Insert(NodeId(6),1);
	implementation.Insert(NodeId(7),0);
	implementation.Insert(NodeId(8),0);

	double f_weight_1_6 = iter++.ReverseInnerProduct();

	if ( f_weight_1_6 != F_WEIGHT_1_6 )
		return false;

	double f_weight_2_6 = iter++.ReverseInnerProduct();

	if ( f_weight_2_6 != F_WEIGHT_2_6 )
		return false;

	double f_weight_3_6 = iter++.ReverseInnerProduct();

	if ( f_weight_3_6 != F_WEIGHT_3_6 )
		return false;

	double f_no_weight_4_6 = iter++.ReverseInnerProduct();

	if ( f_no_weight_4_6 != NO_WEIGHT )
		return false;

	double f_no_weight_5_6 = iter++.ReverseInnerProduct();

	if ( f_no_weight_5_6 != NO_WEIGHT )
		return false;

	if ( iter != implementation.end(0,p_dummy) )
		return false;

	implementation.Insert(NodeId(6),0);
	implementation.Insert(NodeId(7),1);
	implementation.Insert(NodeId(8),0);

	iterator iter2 = implementation.begin(0,p_dummy);
	double f_no_weight_1_7 = iter2++.ReverseInnerProduct();

	if ( f_no_weight_1_7 != NO_WEIGHT )
		return false;

	double f_no_weight_2_7 = iter2++.ReverseInnerProduct();

	if ( f_no_weight_2_7 != NO_WEIGHT )
		return false;

	double f_weight_3_7 = iter2++.ReverseInnerProduct();

	if ( f_weight_3_7 != F_WEIGHT_3_7 )
		return false;


	// etc...

	return true;
}


bool SparseImplementationTest::TestReverseInnerProductLayer2
	(
		D_LayeredReversibleSparseImplementation &implementation
	) const
{

	typedef D_LayeredReversibleSparseImplementation::WeightLayerIterator iterator;
	iterator* p_dummy = 0;
	iterator iter = implementation.begin(1,p_dummy);

	implementation.Insert(NodeId(9), 1);
	implementation.Insert(NodeId(10),0);

	double f_weight_6_9 = iter++.ReverseInnerProduct();

	if ( f_weight_6_9 != F_WEIGHT_6_9 )
		return false;

	double f_weight_7_9 = iter++.ReverseInnerProduct();

	if ( f_weight_7_9 != F_WEIGHT_7_9 )
		return false;

	double f_no_weight_8_9 = iter++.ReverseInnerProduct();

	if ( f_no_weight_8_9 != NO_WEIGHT )
		return false;

	if ( iter != implementation.end(1,p_dummy) )
		return false;

	return true;
}


bool SparseImplementationTest::TestReverseThresholdInnerProductLayer1
	(
		D_LayeredReversibleSparseImplementation& implementation
	) const
{
	typedef D_LayeredReversibleSparseImplementation::WeightLayerIteratorThreshold iterator;

	iterator* p_dummy = 0;
	iterator iter = implementation.begin(0,p_dummy);

	implementation.Insert(NodeId(6),1);
	implementation.Insert(NodeId(7),0);
	implementation.Insert(NodeId(8),0);

	double f_weight_0_6 = iter++.ReverseInnerProduct();

	if ( f_weight_0_6 != F_WEIGHT_0_6 )
		return false;


	double f_weight_1_6 = iter++.ReverseInnerProduct();

	if ( f_weight_1_6 != F_WEIGHT_1_6 )
		return false;

	double f_weight_2_6 = iter++.ReverseInnerProduct();

	if ( f_weight_2_6 != F_WEIGHT_2_6 )
		return false;

	double f_weight_3_6 = iter++.ReverseInnerProduct();

	if ( f_weight_3_6 != F_WEIGHT_3_6 )
		return false;

	double f_no_weight_4_6 = iter++.ReverseInnerProduct();

	if ( f_no_weight_4_6 != NO_WEIGHT )
		return false;

	double f_no_weight_5_6 = iter++.ReverseInnerProduct();

	if ( f_no_weight_5_6 != NO_WEIGHT )
		return false;

	if ( iter != implementation.end(0,p_dummy) )
		return false;

	implementation.Insert(NodeId(6),0);
	implementation.Insert(NodeId(7),1);
	implementation.Insert(NodeId(8),0);

	iterator iter2 = implementation.begin(0,p_dummy);
	double f_weight_0_7 = iter2++.ReverseInnerProduct();
	
	if ( f_weight_0_7 != F_WEIGHT_0_7 )
		return false;

	double f_no_weight_1_7 = iter2++.ReverseInnerProduct();

	if ( f_no_weight_1_7 != NO_WEIGHT )
		return false;

	double f_no_weight_2_7 = iter2++.ReverseInnerProduct();

	if ( f_no_weight_2_7 != NO_WEIGHT )
		return false;

	double f_weight_3_7 = iter2++.ReverseInnerProduct();

	if ( f_weight_3_7 != F_WEIGHT_3_7 )
		return false;

	double f_no_weight_4_7 = iter2++.ReverseInnerProduct();

	if ( f_no_weight_4_7 != NO_WEIGHT )
		return false;

	double f_no_weight_5_7 = iter2++.ReverseInnerProduct();

	if ( f_no_weight_5_7 != NO_WEIGHT )
		return false;

	if ( iter2 != implementation.end(0,p_dummy) )
		return false;

	implementation.Insert(NodeId(6),0);
	implementation.Insert(NodeId(7),0);
	implementation.Insert(NodeId(8),1);

	iterator iter3 = implementation.begin(0,p_dummy);
	double f_weight_0_8 = iter3++.ReverseInnerProduct();

	if ( f_weight_0_8 != F_WEIGHT_0_8 )
		return false;

	double f_no_weight_1_8 = iter3++.ReverseInnerProduct();

	if ( f_no_weight_1_8 !=  NO_WEIGHT )
		return false;

	double f_no_weight_2_8 = iter3++.ReverseInnerProduct();

	if ( f_no_weight_2_8 != NO_WEIGHT )
		return false;

	double f_weight_3_8 = iter3++.ReverseInnerProduct();

	if ( f_weight_3_8 != F_WEIGHT_3_8 )
		return false;

	double f_weight_4_8 = iter3++.ReverseInnerProduct();

	if ( f_weight_4_8 != F_WEIGHT_4_8 )
		return false;

	double f_weight_5_8 = iter3++.ReverseInnerProduct();

	if ( f_weight_5_8 != F_WEIGHT_5_8 )
		return false;

	if ( iter3 != implementation.end(0,p_dummy) )
		return 0;

	return true;
}


bool SparseImplementationTest::TestReverseThresholdInnerProductLayer2
	(
		D_LayeredReversibleSparseImplementation &implementation
	) const
{

	typedef D_LayeredReversibleSparseImplementation::WeightLayerIteratorThreshold iterator;
	iterator* p_dummy = 0;
	iterator iter = implementation.begin(1,p_dummy);

	implementation.Insert(NodeId(9), 1);
	implementation.Insert(NodeId(10),0);

	double f_weight_0_9 = iter++.ReverseInnerProduct();

	if ( f_weight_0_9 != F_WEIGHT_0_9 )
		return false;

	double f_weight_6_9 = iter++.ReverseInnerProduct();

	if ( f_weight_6_9 != F_WEIGHT_6_9 )
		return false;

	double f_weight_7_9 = iter++.ReverseInnerProduct();

	if ( f_weight_7_9 != F_WEIGHT_7_9 )
		return false;

	double f_no_weight_8_9 = iter++.ReverseInnerProduct();

	if ( f_no_weight_8_9 != NO_WEIGHT )
		return false;


	if ( iter != implementation.end(1,p_dummy) )
		return false;

	iterator iter2 = implementation.begin(1,p_dummy);

	implementation.Insert(NodeId(9), 0);
	implementation.Insert(NodeId(10),1);

	double f_weight_0_10 = iter2++.ReverseInnerProduct();

	if ( f_weight_0_10 != F_WEIGHT_0_10 )
		return false;

	double f_no_weight_6_10 = iter2++.ReverseInnerProduct();

	if ( f_no_weight_6_10 != NO_WEIGHT )
		return false;

	double f_weight_7_10 = iter2++.ReverseInnerProduct();

	if ( f_weight_7_10 != F_WEIGHT_7_10 )
		return false;

	double f_weight_8_10 = iter2++.ReverseInnerProduct();

	if ( f_weight_8_10 != F_WEIGHT_8_10 )
		return false;

	if ( iter2 != implementation.end(1,p_dummy) )
		return false;

	return true;
}

bool SparseImplementationTest::ReverseIteratorTest() const
{
	SparseImplementationTest test;

	D_LayeredReversibleSparseImplementation implementation = test.CreateTestReversibleLayeredSparseImplementation<D_ReversibleSparseNode>();
	implementation.InsertReverseImplementation();


	// Iterate over all weights in the network in the stupidest possible way:

	if ( ! TestReverseInnerProductLayer1(implementation) ||
	     ! TestReverseInnerProductLayer2(implementation) )
		 return false;

	return true;
}


bool SparseImplementationTest::ReverseIteratorThresholdTest() const
{
	SparseImplementationTest test;

	D_LayeredReversibleSparseImplementation implementation = test.CreateTestReversibleLayeredSparseImplementation<D_ReversibleSparseNode>();
	implementation.InsertReverseImplementation();


	// Iterate over all weights in the network in the stupidest possible way:

	if (! TestReverseThresholdInnerProductLayer1(implementation) ||
	    ! TestReverseThresholdInnerProductLayer2(implementation) )
		return false;

	// Now create an iterator for the output layer 

	typedef D_LayeredReversibleSparseImplementation::WeightLayerIteratorThreshold iterator;
	iterator* p_dummy = 0;
	iterator iter = implementation.begin(2,p_dummy);

	// This is fine, but a ReverseInnerProduct should be a nono
	try
	{
		iter.ReverseInnerProduct();
	}
	catch (SparseLibIteratorException)
	{
		// do something unimportant
		int i = 0;
		// do something with it , tio suppress compiler warnings
		i++;
	}
	return true;
}


bool SparseImplementationTest::TestSparseNodeStreaming() const
{
	// Careful: the Nodes must be aligned in memory so that input
	// pointers are calculated correctly, hence vector or valarray 
	// is necessary
	vector<D_SparseNode> vector_of_nodes(3);
	vector_of_nodes[0].SetId(NodeId(1));

	vector_of_nodes[1].SetId(NodeId(2));

	vector_of_nodes[2].SetId(NodeId(3));

	D_SparseNode::connection connect1(&vector_of_nodes[1],2);
	D_SparseNode::connection connect2(&vector_of_nodes[2],3);

	vector_of_nodes[0].PushBackConnection(connect1);
	vector_of_nodes[0].PushBackConnection(connect2);

	string file_test = STR_TEST_DIR + STR_NODESTREAMING_TEST;
	ofstream stream_node_test(file_test.c_str());
	if ( ! stream_node_test )
		throw SparseLibTestOpenException();

	stream_node_test << vector_of_nodes[0];
	stream_node_test.close();

	ifstream stream_input_node_test(file_test.c_str());
	if ( ! stream_input_node_test )
		return false;

	vector<D_SparseNode> vector_of_other_nodes(3);
	vector_of_other_nodes[0].SetId(NodeId(1));
	vector_of_other_nodes[1].SetId(NodeId(2));
	vector_of_other_nodes[2].SetId(NodeId(3));

	vector_of_other_nodes[0].FromStream(stream_input_node_test);

	return true;
}


bool SparseImplementationTest::DynamicInsertionTest() const
{
	// Obsolete
	return true;
}


bool SparseImplementationTest::TestImplementationStreaming() const
{
	SparseImplementationTest test;
	D_LayeredReversibleSparseImplementation implementation = test.CreateTestReversibleLayeredSparseImplementation<D_ReversibleSparseNode>();

	string file_test = STR_TEST_DIR + STR_IMPLEMENTATIONSTREAMING_TEST;
	ofstream stream_implementation_test(file_test.c_str());

	if ( ! stream_implementation_test)
		throw SparseLibTestOpenException();


	stream_implementation_test << implementation;
	stream_implementation_test.close();

	ifstream stream_input_implementation_test(file_test.c_str());

	D_LayeredReversibleSparseImplementation implementation_from_stream(stream_input_implementation_test);


	// test a few values
	double f_weight_0_6 = 0.0;
	bool b_weight_result = 
		implementation.GetWeight
		(
			NodeId(6),
			NodeId(0), 
			f_weight_0_6
		);
	if ( f_weight_0_6 != F_WEIGHT_0_6 )
		return false;

	double f_weight_1_6 = 0.0;
	b_weight_result &= 
		implementation.GetWeight
		(
			NodeId(6),
			NodeId(1),
			f_weight_1_6
		);

	if ( f_weight_1_6 != F_WEIGHT_1_6 )
		return false;

	double f_weight_0_9 = 0.0;
	b_weight_result &= 
		implementation.GetWeight
		(
			NodeId(9),
			NodeId(0),
			f_weight_0_9 
		);

	if ( f_weight_0_9 != F_WEIGHT_0_9 )
		return false;

	double f_weight_6_9 = 0.0;
	b_weight_result &= 
		implementation.GetWeight
		(
			NodeId(9),
			NodeId(6),
			f_weight_6_9
		);

	if ( f_weight_6_9 != F_WEIGHT_6_9 )
		return false;

	// Write out the file again, a diff should give no differences
	string string_file_text_copy(STR_TEST_DIR + STR_IMPLEMENTATIONSTREAMING_TEST_COPY);
	ofstream stream_copy(string_file_text_copy.c_str());

	stream_copy << implementation;

	return true;
}
 
bool SparseImplementationTest::PredecessorIteratorTest() const
{
	D_SparseNode node_1, node_2, node_3;

	// set Ids
	node_1.SetId(NodeId(1));
	node_2.SetId(NodeId(2));
	node_3.SetId(NodeId(3));

	// create two connections: 2->1 and 3->1
	double f_weight_plus = 2.0;
	double f_weight_min = -2.0;

	pair<D_SparseNode*,double> connection_12(&node_2, f_weight_plus);

	pair<D_SparseNode*,double> connection_13(&node_3, f_weight_min);

	// add the Connections
	node_1.PushBackConnection(connection_12);
	node_1.PushBackConnection(connection_13);

	// Set activation in Node 2 and 3
	node_2.SetValue(1.0);
	node_3.SetValue(1.0);

	D_SparseNode::predecessor_iterator iter;

	iter = node_1.begin();
	if (iter->MyNodeId() != NodeId(2))
		return false;
	if (iter.GetWeight() != f_weight_plus)
		return false;

	iter++;
	if (iter->MyNodeId() != NodeId(3))
		return false;
	if (iter.GetWeight() != f_weight_min)
		return false;

	iter++;

	int number_nodes = 0;
	for (iter = node_1.begin(); iter != node_1.end(); iter++)
		number_nodes++;

	if (number_nodes != 2)
		return false;
	
	D_SparseNode::predecessor_iterator iter_node3 =
		std::find
		(
			node_1.begin(),
			node_1.end(),
			NodeId(3)
		);
	iter_node3.SetWeight(3.0);

	return true;
}

bool SparseImplementationTest::LayerOrderTest() const {
	// Ordinary SparseNodes
	SparseImplementationTest test;
	D_LayeredSparseImplementation implementation = test.CreateTestLayeredSparseImplementation<D_SparseNode>();

	NodeIterator<D_SparseNode> iter = implementation.begin();
	NodeId id = iter->MyNodeId();
	if (id != NodeId(1) )
		return false;

	iter++;
	iter++;
	iter++;
	iter++;
	iter++;
	iter++;
	iter++;
	iter++;
	iter++;

	// this should be false, because our implementation has 10 Nodes:
	if ( ! (iter != implementation.end()) )
		return false;

	if ( iter->MyNodeId() != NodeId(10) )
		return false;

	// but now it should be equal !
	if ( ++iter != implementation.end() )
		return false;
	else
		return true;
}

bool SparseImplementationTest::SparseImplementationAllocatorTest() const
{
	typedef vector<D_SparseNode, SparseImplementationAllocator<D_SparseNode> > vector_alloc; 
	vector_alloc vec_test;

	// add three nodes
       	D_SparseNode node_1;
	node_1.SetId(NodeId(0));

	D_SparseNode node_2;
	node_2.SetId(NodeId(1));

	D_SparseNode node_3;
	node_3.SetId(NodeId(2));

	vec_test.push_back(node_1);
	vec_test.push_back(node_2);
	vec_test.push_back(node_3);

       	D_SparseNode::connection connection0_2(&vec_test[2], 1.0);
	vec_test[0].PushBackConnection(connection0_2);

	D_SparseNode::connection connection2_0(&vec_test[0], 2.0);
	vec_test[2].PushBackConnection(connection2_0);

	D_SparseNode::connection connection1_1(&vec_test[1], 3.0);
	vec_test[1].PushBackConnection(connection1_1);

	D_SparseNode node_4;
	node_4.SetId(NodeId(4));

	D_SparseNode node_5;
	node_5.SetId(NodeId(5));
	
	vec_test.push_back(node_4);
	vec_test.push_back(node_5);

	D_SparseNode::predecessor_iterator iter_begin = vec_test[0].begin();
	D_SparseNode::predecessor_iterator iter_end = vec_test[0].end();

	return true;
}

bool SparseImplementationTest::NodeVectorTest() const
{
	vector<D_SparseNode, SparseImplementationAllocator<D_SparseNode> > vec_node1(2);
	vec_node1[0].SetId(NodeId(1));
	vec_node1[1].SetId(NodeId(2));
	
	D_SparseNode::connection con(&(vec_node1[1]),55.0);
	vec_node1[0].PushBackConnection(con);


	vector<D_SparseNode, SparseImplementationAllocator<D_SparseNode> > vec_node2;

	vec_node2 = vec_node1;

	return true;
}

bool SparseImplementationTest::ImplementationCreationTest() const
{
	vector<NodeId> empty;

	vector<NodeLink> vec_link;

	NodeLink link(NodeId(1), empty);
	vec_link.push_back(link);

	link = NodeLink(NodeId(2),empty);
	vec_link.push_back(link);

	link = NodeLink(NodeId(3),empty);
	vec_link.push_back(link);

	vector<NodeId> ToFour;
	ToFour.push_back(NodeId(1));
	ToFour.push_back(NodeId(2));
	link = NodeLink(NodeId(4),ToFour);
	vec_link.push_back(link);

	vector<NodeId> ToFive;
	ToFive.push_back(NodeId(2));
	ToFive.push_back(NodeId(3));
	link = NodeLink(NodeId(5),ToFive);
	vec_link.push_back(link);

	// created on the heap, because the Architecture will be owning
	NodeLinkCollection* p_collect = new NodeLinkCollection(vec_link);

	ofstream o_stream((STR_TEST_DIR + STR_IMPLEMENTATION_EXAMPLE).c_str());
	o_stream << *p_collect;
	o_stream.close();


	Architecture arch(p_collect);
	// SparseImplementation< SparseNode<double, double> > is 
	// typedefed to D_SparseImplementation
	D_SparseImplementation imp(&arch);

	return true;
}

bool SparseImplementationTest::LayeredArchitectureTest() const
{
	//create a numerical description of the layers
	vector<Number> vec_layer;

	// a three layered network with 10, 5, 3 nodes
	// in layer 0, 1, 2, respectively
	vec_layer.push_back(10);
	vec_layer.push_back(5);
	vec_layer.push_back(3);

	LayeredArchitecture arch(vec_layer);

	D_LayeredSparseImplementation imp(&arch);

	int n_nodes  = imp.NumberOfNodes();
	int n_layers = imp.NumberOfLayers();

	if ( n_nodes != 18 )
		return false;
	if (n_layers != 3 )
		return false;

	return true;
}

bool SparseImplementationTest::NodeLinkCollectionStreamTest() const
{
	// open an input stream
	ifstream i_stream((STR_TEST_DIR + STR_IMPLEMENTATION_EXAMPLE).c_str());

	if ( ! i_stream )
		return false;

	// construct a NodeLinkCollection from its ASCII representation
	NodeLinkCollection* p_collection = new NodeLinkCollection(i_stream);

	// Create an Architecture from the node link collection
	Architecture arch(p_collection);

	// Create a SparseImplementation
	D_SparseImplementation imp(&arch);

	//
	Number n_nodes = imp.NumberOfNodes();

	// TODO, find out why p_collection->NumberOfNodes gives 0
	if (n_nodes == 0 )
		return false;
	return true;
}

bool SparseImplementationTest::NavigationTest() const
{
	// Create a 3-layered network
	vector<Number> vec_layer(3);
	vec_layer[0] = 10;
	vec_layer[1] =  5;
	vec_layer[2] =  3;

	LayeredArchitecture arch(vec_layer);
	D_LayeredSparseImplementation imp(&arch);

	// we have an implementation
	// now lets loop over its nodes

	for 
	(
		NodeIterator<D_SparseNode> iter = imp.begin();
		iter != imp.end();
		iter++
	)
		{
			// get the NodeId of every node
			NodeId id = iter->MyNodeId();

			// get its activation
			D_SparseNode::ActivityType activation = iter->GetValue();

			// now do something uselss to it, to shut up compiler warnings
			activation += 2.0;
		}

	// now lets get to the predecessors of Node 7
	NodeIterator<D_SparseNode> iter_7 = imp.begin();
	iter_7++; iter_7++; iter_7++; iter_7++; iter_7++; iter_7++; iter_7++;

	// now we can loop over the predecessors of Node 7
	// the node itself has a begin() method for that:
	typedef D_SparseNode::predecessor_iterator pre_iter;
	for
	(
		pre_iter iter_connect = iter_7->begin();
		iter_connect != iter_7->end();
		iter_connect++
	)
		{
			// get the predecessors NodeId
			NodeId id_predecessors = iter_connect->MyNodeId();

			// set the weight of the connection to 10.0
			iter_connect.SetWeight(10.0);
		}

	return true;
}
