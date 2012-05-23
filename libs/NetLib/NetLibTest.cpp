
// Copyright (c) 2005 - 2007 Marc de Kamps
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

#include "NetLibTest.h"
#include "NetLibTestException.h"


#include <iostream>
#include <string>
#include <cmath>

using namespace std;
using namespace NetLib;

NetLibTest::NetLibTest()
{
}


NetLibTest::NetLibTest
(
	boost::shared_ptr<ostream>      stream_log
):
LogStream(stream_log)
{
}

bool NetLibTest::Execute() 
{

	if ( ! ArchitectureTest() )
		return false;
	Record("ArchitectureTest succeeded");

	if ( ! PatternTest() )
		return false;
	Record("PatternTest succeeded");

	return true;
}

bool NetLibTest::LayerIteratorTest() const
{

	return true;
}

bool NetLibTest::ArchitectureTest() const
{
	bool b_return = true;

	LayeredArchitecture architecture = CreateArchitecture(false);

	Number number_of_nodes       = architecture.NumberOfNodes();
	Number number_of_connections = architecture.NumberOfConnections();

	if ( number_of_nodes != 10 || number_of_connections != 11 )
		b_return = false;

	vector<Number> vector_connections = architecture.NumberConnectionsVector();
	if  (
			vector_connections[0] != 7 ||
			vector_connections[1] != 4
		)
		b_return = false;



	LayeredArchitecture architecture_th = CreateArchitecture(true);

	number_of_nodes       = architecture_th.NumberOfNodes();
	number_of_connections = architecture_th.NumberOfConnections();

	if ( number_of_nodes != 10 || number_of_connections != 21 )
		b_return = false;

	vector_connections = architecture_th.NumberConnectionsVector();
	if  (
			vector_connections[0] != 10 ||
			vector_connections[1] != 6
		)
		b_return = false;


	return b_return;
}

LayeredArchitecture
	NetLibTest::CreateArchitecture(bool b_threshold) const
{
	// Tested: 04-08-2003
	// Marc de Kamps
	// A simple sparse layered structure with some irregularities
	//                    *       *
	//                   /  \   /   \
	//                  *      *      *
	//               /  |  \   |    / | \
	//              *   *      *      *    *
	// Create layer vector

	vector<Layer> vector_layer_sizes(3);
	vector_layer_sizes[0] = 5;
	vector_layer_sizes[1] = 3;
	vector_layer_sizes[2] = 2;

	// Create NodeLinkCollection
	vector<NodeLink> vector_of_node_links(0);

	// Input layer
	vector<NodeId> vector_without_inputs(0);
	for (int node_id = 1; node_id <= 5; node_id++)
	{
		NodeId id(node_id);
		NodeLink link(id,vector_without_inputs);
		vector_of_node_links.push_back(link);
	}

	// Middle layer
	vector<NodeId> vector_middle_1(3);
	vector_middle_1[0] = NodeId(1);
	vector_middle_1[1] = NodeId(2);
	vector_middle_1[2] = NodeId(3);
	NodeLink link_middle_1(NodeId(6),vector_middle_1);
	vector_of_node_links.push_back(link_middle_1);

	vector<NodeId> vector_middle_2(1);
	vector_middle_2[0] = NodeId(3);
	NodeLink link_middle_2(NodeId(7),vector_middle_2);
	vector_of_node_links.push_back(link_middle_2);

	vector<NodeId> vector_middle_3(3);
	vector_middle_3[0] = NodeId(3);
	vector_middle_3[1] = NodeId(4);
	vector_middle_3[2] = NodeId(5);
	NodeLink link_middle_3(NodeId(8),vector_middle_3);
	vector_of_node_links.push_back(link_middle_3);
  
	// output layer
	vector<NodeId> vector_out_1(2);
	vector_out_1[0]    = NodeId(6);
	vector_out_1[1]    = NodeId(7);
	NodeLink link_out_1(NodeId(9),vector_out_1);
	vector_of_node_links.push_back(link_out_1);

	vector<NodeId> vector_out_2(2);
	vector_out_2[0]    = NodeId(7);
	vector_out_2[1]    = NodeId(8);
	NodeLink link_out_2(NodeId(10),vector_out_2);
	vector_of_node_links.push_back(link_out_2);

	NodeLinkCollection* p_collection = new NodeLinkCollection(vector_of_node_links);

	// Create Architecture
	LayeredArchitecture
		architecture
		(
			vector_layer_sizes,
			p_collection,
			b_threshold
					
		);


	return architecture;
}

bool NetLibTest::PatternTest() const
{
	D_Pattern pattern(5);

	D_Pattern::pattern_iterator iter_begin = pattern.begin();
	return true;
}

