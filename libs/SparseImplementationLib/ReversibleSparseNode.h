// Copyright (c) 2005 - 2008 Marc de Kamps
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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_REVERSIBLESPARSENODE_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_REVERSIBLESPARSENODE_INCLUDE_GUARD


#include <algorithm>
#include "../UtilLib/UtilLib.h"
#include "SparseNode.h"

using NetLib::SquashingFunctionFactory;
using UtilLib::Index;


namespace SparseImplementationLib
{

	//! ReversibleSparseNode
	//! A ReversibleSparseNode can be used if nodes must know
	//! about their successors, as well as their predecessors

	template <class NodeValue, class WeightValue>
	class ReversibleSparseNode : public AbstractSparseNode<NodeValue,WeightValue>
	{
		typedef AbstractSparseNode<NodeValue,WeightValue>*			node_pointer;
		typedef const AbstractSparseNode<NodeValue,WeightValue>*	const_node_pointer;
		typedef pair<const_node_pointer,const WeightValue*>			reverse_connection;


	public:

		// constructors, destructors and copy operations
		ReversibleSparseNode();
		ReversibleSparseNode(const ReversibleSparseNode&);
		virtual ~ReversibleSparseNode(){}

		ReversibleSparseNode& operator=(const ReversibleSparseNode&);

		// knowledge about successors, in stead of predecesseors
		void InsertReverseConnection(ReversibleSparseNode*);

		Number  NumberOfReverseConnections() const;
		pair<NodeId, WeightValue> SuccessorConnection(Index) const;

		//!
		void Update();

		//! Give the squashing function
		AbstractSquashingFunction& SquashingFunction() const;

		//! Set a new squashing function object and destroy the old one
		void ExchangeSquashingFunction( const AbstractSquashingFunction* );

		double ReverseInnerProduct() const;
		double ReverseThresholdInnerProduct(NodeId, NodeId) const;
	
		// streaming functions
		virtual bool ToStream   (ostream&) const;
		virtual bool FromStream (istream&);

		virtual string Tag() const;

	protected:

		virtual ReversibleSparseNode<NodeValue,WeightValue>* Address(std::ptrdiff_t);

		virtual std::ptrdiff_t Offset(AbstractSparseNode<NodeValue,WeightValue>*) const;


	private:

		void PushBack(const_node_pointer, const WeightValue*); 

		vector<reverse_connection> _vector_of_reverse_connections;

		auto_ptr<AbstractSquashingFunction> _p_function;


	}; // end of ReversibleSparseNode


	typedef ReversibleSparseNode<double,double> D_ReversibleSparseNode;
	// inserted by korbo
	typedef ReversibleSparseNode<float,float> F_ReversibleSparseNode;


} // end of SparseImplementationLib

#endif // include guard 
