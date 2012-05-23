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

#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_ABSTRACTSPARSENODECODE_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_ABSTRACTSPARSENODECODE_INCLUDE_GUARD

#include "AbstractSparseNode.h"

namespace SparseImplementationLib {


	template <class ActivityType, class WeightType>
	AbstractSparseNode<ActivityType,WeightType>::AbstractSparseNode(NodeId id):
	_activation_value(0),
	_my_id(id),
	_vector_of_connections(0)
	{
	}

	template <class ActivityType, class WeightType>
	AbstractSparseNode<ActivityType, WeightType>::AbstractSparseNode
	(
		const AbstractSparseNode<ActivityType, WeightType>& rhs
	):
	_activation_value(rhs._activation_value),
	_my_id(rhs._my_id),
	_vector_of_connections(rhs._vector_of_connections)
	{
	}

	template <class ActivityType, class WeightType>
	AbstractSparseNode<ActivityType,WeightType>&
		AbstractSparseNode<ActivityType,WeightType>::operator=
		(
			const AbstractSparseNode<ActivityType,WeightType>& rhs
		)
	{
		if ( this == &rhs)
			return *this;

		_my_id                 = rhs._my_id;
		_activation_value      = rhs._activation_value;
		_vector_of_connections = rhs._vector_of_connections;

		return *this;
	}

	template <class ActivityType, class WeightType>
	AbstractSparseNode<ActivityType,WeightType>::~AbstractSparseNode()
	{
	}

	template <class ActivityType,class WeightType>
	istream& operator>> //namespace qualifier removed by Johannes
	(
		istream& s, 
		AbstractSparseNode<ActivityType,WeightType>& node
	)
	{
		node.FromStream(s);
		return s;
	}

	template <class ActivityType, class WeightType>
	inline void AbstractSparseNode<ActivityType,WeightType>::SetValue(ActivityType f_value)
	{
		_activation_value = f_value;
	}

	template <class ActivityType, class WeightType>
	inline void AbstractSparseNode<ActivityType,WeightType>::SetId(NetLib::NodeId nid)
	{
		_my_id = nid;
	}

	template <class ActivityType, class WeightType>
	inline ActivityType AbstractSparseNode<ActivityType,WeightType>::GetValue() const
	{
		return _activation_value;
	}

	template <class NodeValue, class WeightValue>
	inline void AbstractSparseNode<NodeValue,WeightValue>::Clear()
	{
		_vector_of_connections.clear();
	}
	

	template <class ActivityType, class WeightType>
	bool AbstractSparseNode<ActivityType,WeightType>::FromStream(istream& s)
	{
		Number n_input;
		string str;
		s >> str;

		if (str != this->Tag())
			throw  SparseLibException("Header error");

		s >> _my_id._id_value >> _activation_value >> n_input;

		for (Number n_input_index = 0; n_input_index < n_input; n_input_index++ )
		{
			// loop over all connections

			int		predec_id_value;// id of this neuron's predecessor
			WeightType	f_weight_value;	// weight value of this connection

			s >> predec_id_value >> f_weight_value;
			connection con_current;

			// relies on correct overload of Address for derived types
			AbstractSparseNode<ActivityType, WeightType>* 
				p_predecessor = this->Address( predec_id_value - MyNodeId()._id_value);


			if (!  p_predecessor )
				return false;
			con_current.first  = p_predecessor;
			con_current.second = f_weight_value;
			PushBackConnection(con_current);
		}

		s >> str;
		if (str != ToEndTag(this->Tag()))
			throw SparseLibException("Header error");
	
		return true;
	}

	template  <class ActivityType, class WeightType>
	bool AbstractSparseNode<ActivityType,WeightType>::ToStream(ostream& s)  const
	{

		Number number_connections = static_cast<Number>( _vector_of_connections.size() );
		s << this->Tag();
		s << " " << _my_id._id_value << " " << _activation_value << " " << number_connections << "\n";


		copy
		(
			_vector_of_connections.begin(),
			_vector_of_connections.end(),
			ostream_iterator<connection>(s," ")
		);

		s << "\n";


		s << ToEndTag(this->Tag()) << "\n";

		return true;
	}	
	
	template <class ActivityType, class WeightType>
	string AbstractSparseNode<ActivityType, WeightType>::Tag() const
	{
		return TAG_ABSTRACTNODE;
	}

	template <class A, class W>
	SparsePredecessorIterator<AbstractSparseNode<A,W> > AbstractSparseNode<A,W>::begin() 
	{

		return ( 
				(_vector_of_connections.begin() != _vector_of_connections.end() ) ?
				SparsePredecessorIterator<AbstractSparseNode<A,W> >(&( *_vector_of_connections.begin()) ):
				SparsePredecessorIterator<AbstractSparseNode<A,W> >(0)
			);
	}

	template <class A, class W>
	SparsePredecessorIterator<AbstractSparseNode<A,W> > AbstractSparseNode<A,W>::end()
	{
		// Modified: 27-07-2006 (MdK)
		// Originally attempted to dereference _vctor_of_connections.end(), which worked under some compilers
		// but which is not legal

		if ( _vector_of_connections.begin() == _vector_of_connections.end() )
			return SparsePredecessorIterator<AbstractSparseNode<A,W> >(0);

		else {

			pair<AbstractSparseNode<A,W>*, WeightType>* p_end = 
				&(* _vector_of_connections.begin() ) + _vector_of_connections.size();

			return SparsePredecessorIterator<AbstractSparseNode<A,W> >(p_end);
		}
	}

	template <class A, class W>
	ConstSparsePredecessorIterator<AbstractSparseNode<A,W> > AbstractSparseNode<A,W>::begin() const 
	{
		// Purpose: const iterator versions are necessary in ClamLib
		// Author: Marc de Kamps
		// Date: 12-12-2006

		return ( 
				(_vector_of_connections.begin() != _vector_of_connections.end() ) ?
				ConstSparsePredecessorIterator<AbstractSparseNode<A,W> >(&( *_vector_of_connections.begin()) ):
				ConstSparsePredecessorIterator<AbstractSparseNode<A,W> >(0)
			);
	}

	template <class A, class W>
	ConstSparsePredecessorIterator<AbstractSparseNode<A,W> > AbstractSparseNode<A,W>::end() const
	{
		// Purpose: const iterator versions are necessary in ClamLib
		// Author: Marc de Kamps
		// Date: 12-12-2006
		if ( _vector_of_connections.begin() == _vector_of_connections.end() )
			return ConstSparsePredecessorIterator<AbstractSparseNode<A,W> >(0);

		else {

			pair<AbstractSparseNode<A,W>*, WeightType>* p_end = 
				&(* _vector_of_connections.begin() ) + _vector_of_connections.size();

			return ConstSparsePredecessorIterator<AbstractSparseNode<A,W> >(p_end);
		}
	}

	template <class NodeValue, class WeightValue>
	inline void AbstractSparseNode<NodeValue, WeightValue>::ApplyOffset
	(
		const_node_pointer p_original_node
	)  const
	{
		// Purpose: after a copy operation the pointer to the Nodes in the connections still point to the old array.
		// With the old start pointer array, one can calculate the offset, and use this with the pointer to the current
		// start array to calculate the new pointers
		// Author: Marc de Kamps
		// Date: 21-02-2006

		typename vector<connection>::iterator it_begin = _vector_of_connections.begin();
		typename vector<connection>::iterator it_end   = _vector_of_connections.end();

		for( typename vector<connection>::iterator iter = it_begin; iter != it_end; iter++ )
		{

			// relies on correct overload of offset to calculate the new value of the pointer which is now in iter->first
			// derived types are larger

			std::ptrdiff_t i_offset = p_original_node->Offset(iter->first);

			// now that we know what the offset is, the derived type can compute its address
			// again, this relies on a correct overload by the derived type

			iter->first = const_cast<node_pointer>(this)->Address(i_offset);

			// the const cast is necessary: this function is called from the copy constructor
			// this function does nothing with the objected pointed to be iter->first, but
			// the pointer may not be const

		}
	}


	template <class NodeValue, class WeightValue>
	inline bool AbstractSparseNode<NodeValue,WeightValue>::IsInputNodeInRange
	(
		const_node_pointer p_begin,
		const_node_pointer p_end   
	) const
	{
		// Do not assume that sizeof(*p_begin) == sizeof(SparseNode)
		// They may be derived objects
		// Therefore just a check if they are in the right range

		for
		(
		   typename vector<connection>::const_iterator iter = _vector_of_connections.begin(); 
		   iter != _vector_of_connections.end(); 
		   iter++
		 )
			if ( iter->first > p_end || iter->first < p_begin )
				return false;

		return true;
	}

	template <class NodeValue, class WeightValue>
	inline void AbstractSparseNode<NodeValue,WeightValue>::ScaleWeights(WeightType scale)
	{
		for 
		(
			typename vector<connection>::iterator iter = _vector_of_connections.begin();
			iter != _vector_of_connections.end();
			iter++
		)
			iter->second *= scale;
	}


	template <class NodeValue, class WeightValue>
	inline void AbstractSparseNode<NodeValue,WeightValue>::PushBackConnection(const connection& con)
	{
		_vector_of_connections.push_back(con);
	}

	template <class NodeValue, class WeightValue>
	inline double AbstractSparseNode<NodeValue,WeightValue>::InnerProduct() const
	{
		return inner_product
			(
				_vector_of_connections.begin(),
				_vector_of_connections.end(),
				_vector_of_connections.begin(),
				0.0,
				plus<WeightValue>(),
				ConnectionProduct<NodeValue,WeightValue>()
			);	

	}

	//! All derived classes from AbstractSpareNode can use operator<<
	template <class ActivityType, class WeightType>
	ostream& operator<<
	(
		ostream& s, 
		const AbstractSparseNode<ActivityType,WeightType>& node)
	{
		node.ToStream(s);
		return s;
	}

}



#endif //include guard
