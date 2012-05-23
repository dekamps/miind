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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEPREDECESSORITERATOR_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEPREDECESSORITERATOR_INCLUDE_GUARD

#include <iterator>

using namespace std;

namespace SparseImplementationLib
{
	//! forward declaration AbstractSparseNode
	template <class NodeValue, class WeightValue> class AbstractSparseNode;

	template <class NodeType> class SparsePredecessorIterator;

	template <class NodeType>
	inline bool operator!=
	(
		const SparsePredecessorIterator<NodeType>& iter_left,
		const SparsePredecessorIterator<NodeType>& iter_right
	);

	//! SparsePredecessorIterator
	//! A SparsePredecessorItertor is provided by an AbstractSparseNode as a means to loop over the
	//! connections that are made to it. A Connection is a pointer-weight combination, where the pointer
	//! points to the AbstractSparseNode's predecessor node and the weight is the weight of the corresponding 
	//! connection.

	template <class NodeType>
	class SparsePredecessorIterator : public iterator<forward_iterator_tag, NodeType >
	{
	public:

		typedef NodeType*						NodePointer;
		typedef typename NodeType::WeightType	WeightType;
		typedef pair<NodePointer,WeightType>	Connection;
		
		SparsePredecessorIterator():_p_current_connection(0){}
		SparsePredecessorIterator(Connection* p_connection):_p_current_connection(p_connection){}

		//! Define integer offsets for the iterator
		SparsePredecessorIterator  operator+ ( int i)
		{
			return _p_current_connection + i;
		}

		//! Define the customary iterator increment (prefix)
		SparsePredecessorIterator& operator++()	
		{ 	
			// prefix 
			++_p_current_connection;
			return *this;
		}

		//! Define the customary iterator increment (postfix)
		SparsePredecessorIterator  operator++(int)	
		{
			// postfix
			SparsePredecessorIterator iter = *this;
			++*this;
			return iter;	
		}

		//! Dereference operator
		NodeType& operator* () const { return *_p_current_connection->first; }

		//! Dereference operator
		NodeType* operator->() const { return _p_current_connection->first; }

		Connection* ConnectionPointer() const { return _p_current_connection; }
	
		//! Provide direct reading access to the weight that corresponds to a connection
		const WeightType& GetWeight()  const { return _p_current_connection->second; }

		//! Provide direct writing access to the weight that corresponds to a connection
		void SetWeight(WeightType weight) const { _p_current_connection->second = weight; }


		friend  
			bool operator!= <>
			(
				const SparsePredecessorIterator<NodeType>&,
				const SparsePredecessorIterator<NodeType>&
			);

	private:

		Connection* _p_current_connection;	
	};


	//! operator!=
	template <class NodeType>
	bool operator!=
	(
		const SparsePredecessorIterator<NodeType>& iter_left,
		const SparsePredecessorIterator<NodeType>& iter_right
	)
	{ 
		// this works because each predecessor is guaranteed to occur
		// only once
		return ( iter_left._p_current_connection != iter_right._p_current_connection);
	}

	//! operator==
	template <class NodeType>
	inline bool operator==
	(
		const SparsePredecessorIterator<NodeType>& iter_left,
		const SparsePredecessorIterator<NodeType>& iter_right
	)
	{
		return ! (iter_left != iter_right) ;
	}

	template <class NodeType>
	inline Number operator-
	(
		const SparsePredecessorIterator<NodeType>& iter_first,
		const SparsePredecessorIterator<NodeType>& iter_second
	);

} // end of SparseImplementationLib

#endif // include guard
