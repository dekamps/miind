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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_CONSTSPARSEPREDECESSORITERATOR_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_CONSTSPARSEPREDECESSORITERATOR_INCLUDE_GUARD

#include <iterator>

using namespace std;

namespace SparseImplementationLib
{
	//! forward declaration 
	template <class NodeValue, class WeightValue> class AbstractSparseNode;

	template <class NodeType> class SparsePredecessorIterator;
	template <class NodeType>
	inline bool operator!=
	(
		const SparsePredecessorIterator<NodeType>& iter_left,
		const SparsePredecessorIterator<NodeType>& iter_right
	);


	template <class NodeType> class ConstSparsePredecessorIterator; 
	template <class NodeType> bool operator!= 
	(
	        const ConstSparsePredecessorIterator<NodeType>&,
	        const ConstSparsePredecessorIterator<NodeType>&
	);
	
	//! SparsePredecessorIterator

	template <class NodeType>
	class ConstSparsePredecessorIterator : public iterator<forward_iterator_tag, NodeType >
	{
	public:

		typedef NodeType*						NodePointer;
		typedef typename NodeType::WeightType	WeightType;
		typedef pair<NodePointer,WeightType>	Connection;
		
		ConstSparsePredecessorIterator():_p_current_connection(0){}
		ConstSparsePredecessorIterator(Connection* p_connection):_p_current_connection(p_connection){}

		ConstSparsePredecessorIterator  operator+ ( int );
		ConstSparsePredecessorIterator& operator++()	
		{ 	// prefix 
				++_p_current_connection;
				return *this;
		}

		ConstSparsePredecessorIterator  operator++(int)	
		{
			// postfix
			ConstSparsePredecessorIterator iter = *this;
			++*this;
			return iter;	
		}

		const NodeType& operator* () const { return *_p_current_connection->first; }

		const NodeType* operator->() const { return _p_current_connection->first; }

		Connection* ConnectionPointer() const { return _p_current_connection; }
	
		const WeightType& GetWeight()  const { return _p_current_connection->second; }

		void SetWeight(WeightType weight) const { _p_current_connection->second = weight; }


		friend  bool operator!= <>
			(
				const ConstSparsePredecessorIterator<NodeType>&,
				const ConstSparsePredecessorIterator<NodeType>&
			);

	private:

		Connection* _p_current_connection;	
	};


	//! operator!=
	template <class NodeType>
	bool operator!=
	(
		const ConstSparsePredecessorIterator<NodeType>& iter_left,
		const ConstSparsePredecessorIterator<NodeType>& iter_right
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
		const ConstSparsePredecessorIterator<NodeType>& iter_left,
		const ConstSparsePredecessorIterator<NodeType>& iter_right
	)
	{
		return ! (iter_left != iter_right) ;
	}

	template <class NodeType>
	inline Number operator-
	(
		const ConstSparsePredecessorIterator<NodeType>& iter_first,
		const ConstSparsePredecessorIterator<NodeType>& iter_second
	);

} // end of SparseImplementationLib

#endif // include guard
