// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_COLLECTSUCCESORINFORMATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_COLLECTSUCCESORINFORMATIONCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "CollectSuccesorInformation.h"

using NetLib::NodeLinkCollection;

namespace SparseImplementationLib
{

		template <class NodeType>
		CollectSuccesorInformation<NodeType>::CollectSuccesorInformation
		(
			bool b_threshold,
			const NodeLinkCollection& reverse_collection, 
			vector<NodeType, SparseImplementationAllocator<NodeType> >& vector_of_nodes, 
			int number_of_input_nodes
		):
		_b_threshold			    (b_threshold),
		_number_of_input_nodes      (number_of_input_nodes),
		_ref_vector_of_nodes        (vector_of_nodes),
		_ref_reverse_link_collection(reverse_collection)
		{
		}

		template <class NodeType>
		NodeType CollectSuccesorInformation<NodeType>::operator ()(NodeType& node) const
		{
	
			// In a NodeLinkCollection NodeId(0), the threshold Node, is not
			// represented explictely, so it has to be handled separately
			if ( node.MyNodeId() != NodeId(0))
				return HandleOrdinaryNode(node);
			else
				return HandleThresholdNode(node);
		}

		template <class NodeType>
		NodeType CollectSuccesorInformation<NodeType>::HandleOrdinaryNode(NodeType& node) const
		{
			NodeType node_return = node;

			// Get the list of 'successors'
			NodeLink link_successors = _ref_reverse_link_collection[node.MyNodeId()._id_value - 1];

			// Test to see if this is really the Id we want:
			if ( link_successors.MyNodeId() != node.MyNodeId() )
				throw NetLibException(string("Unexpected NodeId"));

			// We are fine, we have the right link
			// Now loop over all successors
			Number number_of_successors = link_successors.Size();
			for (Index index_successor = 0; index_successor < number_of_successors; index_successor++)
			{
				NodeId id_successor = link_successors[index_successor];
				node_return.InsertReverseConnection(&_ref_vector_of_nodes[id_successor._id_value]);
			}

			return node_return;
		}

		template <class NodeType>
		NodeType CollectSuccesorInformation<NodeType>::HandleThresholdNode(NodeType& node) const
		{
			// Handle the following issue:
			// All Nodes have the Threhold Node (id 0) as predecessor, if a _threshold
			// architecture is specified
			if ( _b_threshold )
			{
				NodeType node_return = node;
				Number number_of_nodes = static_cast<Number>(_ref_vector_of_nodes.size());
		
				//NodeId(0) is not a predecessor of itself
				for (Number number = 1; number < number_of_nodes; number++)
					node_return.InsertReverseConnection( &(_ref_vector_of_nodes[number]) );

				return node_return;
			}
			else
				return node;
		}

} // end of SparseImplementationLib

#endif // include guard
