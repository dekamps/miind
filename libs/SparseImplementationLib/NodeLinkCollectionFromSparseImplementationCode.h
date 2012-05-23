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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_NODELINKCOLLECTIONFROMSPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_NODELINKCOLLECTIONFROMSPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD


#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "NodeLinkCollectionFromSparseImplementation.h"


namespace SparseImplementationLib
{
		template <class NodeType>
		NodeLinkCollectionFromSparseImplementation<NodeType>::NodeLinkCollectionFromSparseImplementation()
		{
		}

		template <class NodeType>
		vector<NodeLink> NodeLinkCollectionFromSparseImplementation<NodeType>::CreateNodeLinkVector
		(
			const vector<NodeType, SparseImplementationAllocator<NodeType> >& vector_of_nodes
		)
		{
			// for every Node in the SparseImplementation, there will be a NodeLink
			vector<NodeLink> vector_node_link;

			// In the SparseNode array, NodeId(0) is also present, but it should not go into the NodeLinkCollection
			assert(vector_of_nodes.size() > 0);
			Number number_of_nodes = static_cast<int>(vector_of_nodes.size()) - 1;

			// but we do the structure later
			vector<NodeId> vector_node_empty(0);

			for (Number i_node_id = 0; i_node_id < number_of_nodes; i_node_id++)
			{
				// remember threshold neuron
				NodeLink link(NodeId( static_cast<int>(i_node_id+1) ),vector_node_empty);
				vector_node_link.push_back(link);
			}
			return vector_node_link;
		}

		template <class NodeType>
		void NodeLinkCollectionFromSparseImplementation<NodeType>::AssignReversedRelation
		(
			vector<NodeLink>& vector_of_node_links, 
			const vector<NodeType, SparseImplementationAllocator<NodeType> >& vector_of_nodes
		)
		{
			// copy vector_of_node, this is inefficient,
			// but it is assumed that reversing networks is a one time operation

			vector<NodeType, SparseImplementationAllocator<NodeType> > vector_nodes = vector_of_nodes;

			for
			(
				typename vector<NodeType, SparseImplementationAllocator<NodeType> >::iterator iter_node = vector_nodes.begin();
				iter_node != vector_nodes.end();
				iter_node++
			)
			{
				for 
				(
					typename AbstractSparseNode<typename NodeType::ActivityType,typename NodeType::WeightType>::predecessor_iterator iter_predecessor = iter_node->begin();
					iter_predecessor != iter_node->end();
					iter_predecessor++
				)
				{
					//compensate the offset of the lacking threshold Node
					assert(iter_predecessor->MyNodeId()._id_value < static_cast<int>(vector_of_node_links.size()) );

					int index = iter_predecessor->MyNodeId()._id_value-1;
					if ( index >= 0 )
						vector_of_node_links[index].PushBack
						(
							NodeId
							( 
								static_cast<int>(iter_node - vector_nodes.begin())
							) 
						);
					// else
						// do nothing
				}
			}

		}

		template <class NodeType>
		NodeLinkCollection NodeLinkCollectionFromSparseImplementation<NodeType>::CreateReverseNodeLinkCollection
		(
			const vector<NodeType, SparseImplementationAllocator<NodeType> >& vector_of_nodes
		)
		{
			// create as many NodeLinks as there are Nodes 
			vector<NodeLink> vector_of_node_links =  CreateNodeLinkVector(vector_of_nodes);

			AssignReversedRelation
			(
				vector_of_node_links,
				vector_of_nodes
			);

			NodeLinkCollection collection_return(vector_of_node_links);

			return collection_return;
		}

} // end of sparseLibImplementation

#endif // include guard
