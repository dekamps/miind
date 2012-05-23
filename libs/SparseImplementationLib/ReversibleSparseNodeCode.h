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
#ifndef _CODE_LIBS_NETLIB_REVERSIBLESPARSENEURONCODE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_REVERSIBLESPARSENEURONCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "ReversibleSparseNode.h"


namespace SparseImplementationLib
{
		
		template <class NodeValue, class WeightValue>
		ReversibleSparseNode<NodeValue,WeightValue>::ReversibleSparseNode():
		AbstractSparseNode<NodeValue,WeightValue>(NodeId(0)),
		_p_function(new NoSquashingFunction)
		{
		}

		template <class NodeValue, class WeightValue>
		ReversibleSparseNode<NodeValue,WeightValue>::ReversibleSparseNode(const ReversibleSparseNode<NodeValue,WeightValue>& rhs):
		AbstractSparseNode<NodeValue,WeightValue>(rhs),
		_vector_of_reverse_connections(rhs._vector_of_reverse_connections),
		_p_function(auto_ptr<AbstractSquashingFunction>(rhs._p_function->Clone()))
		{
		}

		template <class NodeValue, class WeightValue>
		ReversibleSparseNode<NodeValue,WeightValue>&
			ReversibleSparseNode<NodeValue,WeightValue>::operator=(const ReversibleSparseNode<NodeValue,WeightValue>& rhs)
		{
			if (&rhs == this)
				return *this;

			AbstractSparseNode<NodeValue,WeightValue>::operator=(rhs);

			_vector_of_reverse_connections = rhs._vector_of_reverse_connections;

			_p_function = auto_ptr<AbstractSquashingFunction>(rhs._p_function->Clone());
			return *this;
		}

		template <class NodeValue, class WeightValue>
		string ReversibleSparseNode<NodeValue,WeightValue>::Tag() const
		{
			return TAG_REVERSIBLE;
		}
		
		template <class NodeValue, class WeightValue>
		bool ReversibleSparseNode<NodeValue,WeightValue>::ToStream(ostream& s) const
		{
			// For the moment only call the base class
			AbstractSparseNode<NodeValue,WeightValue>::ToStream(s);

			_p_function->ToStream(s);

			return true;
		}

		template <class NodeValue, class WeightValue>
		ReversibleSparseNode<NodeValue,WeightValue>* ReversibleSparseNode<NodeValue,WeightValue>::Address(std::ptrdiff_t index)
		{
			return (this + index);
		}

		template <class NodeValue, class WeightValue>
		bool ReversibleSparseNode<NodeValue,WeightValue>::FromStream(istream& s) 
		{

			// For the moment only call the base class
			AbstractSparseNode<NodeValue,WeightValue>::FromStream(s);

			SquashingFunctionFactory factory;
			_p_function = factory.FromStream(s);

			return true;
		}

		template <class NodeValue, class WeightValue>
		void ReversibleSparseNode<NodeValue,WeightValue>::InsertReverseConnection
		(
			ReversibleSparseNode<NodeValue,WeightValue>* p_successor
		)
		{
			// We already have the pointer to my successor,
			// We need to know about the weight

			// and I can ask my successor about its weight from me to it

			typename AbstractSparseNode<NodeValue,WeightValue>::predecessor_iterator iter_predecessor = 
				std::find
				(
					p_successor->begin(),
					p_successor->end(),
					this->MyNodeId()
				);

			// something is wrong if i can't find myself at my predecessor
			if ( iter_predecessor != p_successor->end() )
			{
				const WeightValue* p_weight = &(iter_predecessor.GetWeight());

				// we're set
				PushBack(p_successor,p_weight);
			}
			else
				throw NetLibException(STRING_PREDECESSOR_NOT_FOUND);


		}

		template <class NodeValue, class WeightValue>
		void ReversibleSparseNode<NodeValue,WeightValue>::PushBack
		(
			const_node_pointer p_node, 
			const WeightValue* p_weight
		)
		{
			_vector_of_reverse_connections.push_back(reverse_connection(p_node,p_weight));
		}

		template <class NodeValue, class WeightValue>
		Number ReversibleSparseNode<NodeValue,WeightValue>::NumberOfReverseConnections() const
		{
			return static_cast<Number>( _vector_of_reverse_connections.size() );
		}

		template <class NodeValue, class WeightValue>
		pair<NodeId,WeightValue> ReversibleSparseNode<NodeValue,WeightValue>::SuccessorConnection(Index index) const
		{
			pair<NodeId,WeightValue> pair_return;
			pair_return.first  = _vector_of_reverse_connections[index].first->MyNodeId();
			pair_return.second = *_vector_of_reverse_connections[index].second;

			return pair_return;
		}

		template <class NodeValue, class WeightValue>
		double ReversibleSparseNode<NodeValue,WeightValue>::ReverseInnerProduct() const
		{
			typename vector<reverse_connection>::const_iterator iter_begin = _vector_of_reverse_connections.begin();
			typename vector<reverse_connection>::const_iterator iter_end   = _vector_of_reverse_connections.end();

			double f_result = 0.0;
			for(typename vector<reverse_connection>::const_iterator iter = iter_begin; iter != iter_end; iter++)
			{
				f_result += ( iter->first->GetValue() )*( *(iter->second) );
			}
			return f_result;
		}

		template <class NodeValue, class WeightValue>
		void ReversibleSparseNode<NodeValue,WeightValue>::Update()
		{
			this->SetValue( (*_p_function.get())(this->InnerProduct()) );
		}

		template <class NodeValue, class WeightValue>
		void ReversibleSparseNode<NodeValue,WeightValue>::ExchangeSquashingFunction
		( 
			const AbstractSquashingFunction* p_function 
		)
		{
			_p_function = auto_ptr<AbstractSquashingFunction>(p_function->Clone());
		}

		template <class NodeValue, class WeightValue>
		AbstractSquashingFunction& ReversibleSparseNode<NodeValue,WeightValue>::SquashingFunction() const
		{
			return *_p_function;
		}


		template <class NodeValue, class WeightValue>
		double ReversibleSparseNode<NodeValue,WeightValue>::ReverseThresholdInnerProduct
		(
			NetLib::NodeId id_start_layer, 
			NetLib::NodeId id_end_layer
		) const
		{
		        assert (this->MyNodeId() == NetLib::THRESHOLD_ID); 
			typename vector<reverse_connection>::const_iterator iter_begin = _vector_of_reverse_connections.begin();
			int i_offset     = id_start_layer._id_value - iter_begin->first->MyNodeId()._id_value;
			int i_difference = id_end_layer._id_value   - id_start_layer._id_value;
			iter_begin += i_offset;

			// Remember: end iterator has to point past last element
			typename vector<reverse_connection>::const_iterator iter_end   = iter_begin + i_difference + 1;

			double f_result = 0.0;
			for( typename vector<reverse_connection>::const_iterator iter = iter_begin; 
				 iter != iter_end; 
				 iter++
				)
			{
				f_result += ( iter->first->GetValue() )*( *(iter->second) );
			}
			return f_result;
		}

		template<class NodeValue, class WeightValue>
		std::ptrdiff_t ReversibleSparseNode<NodeValue,WeightValue>::Offset(AbstractSparseNode<NodeValue,WeightValue>* p_abstract_node) const
		{
			ReversibleSparseNode<NodeValue,WeightValue>* p_node = dynamic_cast<ReversibleSparseNode<NodeValue,WeightValue>*>(p_abstract_node);
			if (! p_node)
				throw SparseLibException(OFFSET_ERROR);

			return (p_node - this);
		}


} // end of SparseImplementationLib

#endif // include guard
