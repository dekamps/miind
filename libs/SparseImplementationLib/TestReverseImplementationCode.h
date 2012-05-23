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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_TESTREVERSEIMPLEMENTATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_TESTREVERSEIMPLEMENTATIONCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "TestReverseImplementation.h"


namespace SparseImplementationLib
{

		template <class NodeType, class WeightValue>
		TestReverseImplementation<NodeType,WeightValue>::TestReverseImplementation
		(
			vector<NodeType, SparseImplementationAllocator<NodeType> >& vector_of_nodes
		):
		_ref_vector_of_nodes(vector_of_nodes)
		{
		}

		template <class NodeType, class WeightValue>
		void TestReverseImplementation<NodeType, WeightValue>::operator ()(NodeType& node) 
		{
			//TODO: need successor iterators !
			for
			(
				typename NodeType::predecessor_iterator iter_predecessor = node.begin();
				iter_predecessor != node.end();
				iter_predecessor++
			)
			{
				pair<NodeId,WeightValue> predecessor; 
				predecessor.first  = iter_predecessor->MyNodeId();
				predecessor.second = iter_predecessor.GetWeight();

			
				// Now can we find this Node, as successor in its Predecessor node ?

				if (IsSuccessorInPredecessor(node,predecessor))
					continue;
				else
					throw ReverseImplementationException(string("my successor is not in my predecessor"));
			}

			// Generate a list of predecessor-weight pairs for this node

/*			size_t number_of_predecessors = node.NumberOfInputs();
			for (Index index = 0; index < number_of_predecessors; index++)
			{
				pair<NodeId,WeightValue> predecessor;
				predecessor.first  = node.IdPredecessor(index);
				predecessor.second = node.RetrieveWeight(index);
			
				// Now can we find this Node, as successor in its Predecessor node ?

				if (IsSuccessorInPredecessor(node,predecessor))
					continue;
				else
					throw ReverseImplementationException(string("my successor is not in my predecessor"));
			}*/
		}

		template <class NodeType, class WeightValue>
		bool TestReverseImplementation<NodeType, WeightValue>::IsSuccessorInPredecessor(const NodeType& node, const pair<NodeId,WeightValue>& predecessor) const
		{
			// Establish a predecessor of this Node
			const NodeType& node_predecessor = _ref_vector_of_nodes[predecessor.first._id_value];

			// Look among it successors, if this Node is in
			Number number_of_successors = node_predecessor.NumberOfReverseConnections();
			int index_successor = -1;
			for (int index = 0; index < static_cast<int>(number_of_successors); index++)
			{
				pair<NodeId,WeightValue> successor_pair = node_predecessor.SuccessorConnection(index);
				if  ( successor_pair.first  == node.MyNodeId() &&
					  successor_pair.second == predecessor.second )
				{
					index_successor = index;
					break;
				}
			}

			if (index_successor == -1)
				throw ReverseImplementationException(string("have to figure this one out"));
			else
				return true;
		}

} // end of SparseImplementationLib

#endif  //include guard

