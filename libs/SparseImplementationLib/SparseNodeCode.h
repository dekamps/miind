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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSENEURONIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSENEURONIMPLEMENTATION_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <float.h>
#include <numeric>
#include "../UtilLib/UtilLib.h"
#include "../NetLib/NetLib.h"
#include "ConnectionProduct.h"
#include "SparseNode.h"
#include "LocalDefinitions.h"

using std::inner_product;

namespace SparseImplementationLib
{

	template <class NodeValue, class WeightValue>
	inline SparseNode<NodeValue,WeightValue>::SparseNode():
	  AbstractSparseNode<NodeValue,WeightValue>(NodeId(0)),
	  _p_function(new NoSquashingFunction)
	{
	}

	template <class NeuronValue, class WeightValue>
	inline SparseNode<NeuronValue,WeightValue>::~SparseNode()
	{
	}

	template <class NeuronValue, class WeightValue>
	inline SparseNode<NeuronValue,WeightValue>::SparseNode
	(
		const SparseNode<NeuronValue,WeightValue>& rhs 
	):
	  AbstractSparseNode<NodeValue,WeightValue>(rhs),
	_p_function( auto_ptr<AbstractSquashingFunction>( rhs._p_function->Clone()) )
	{
	}

	template <class NeuronValue, class WeightValue>
	SparseNode<NeuronValue,WeightValue>& 
		SparseNode<NeuronValue,WeightValue>::operator=
			(
				const SparseNode<NeuronValue,WeightValue>& rhs
			)
	{

		if ( this == &rhs )
			return *this;

		AbstractSparseNode<NodeValue,WeightValue>::operator=(rhs);

		_p_function = auto_ptr<AbstractSquashingFunction>( rhs._p_function->Clone() );

		return *this;
	}

	template <class NodeValue, class WeightValue>
	AbstractSquashingFunction& SparseNode<NodeValue,WeightValue>::SquashingFunction() const
	{
		return *_p_function;
	}


	template <class NodeValue, class WeightValue>
	void SparseNode<NodeValue,WeightValue>::Update()
	{
		SetValue( (*_p_function.get())(this->InnerProduct()) );
	}


	template <class NodeValue, class WeightValue>
	void SparseNode<NodeValue,WeightValue>::ExchangeSquashingFunction
	( 
		const AbstractSquashingFunction* p_function 
	)
	{
		_p_function = auto_ptr<AbstractSquashingFunction>(p_function->Clone());
	}

	template <class NodeValue,class WeightValue>
	bool SparseNode<NodeValue,WeightValue>::ToStream(ostream& s) const
	{
	
		NodeValue node_activation = this->GetValue();
		if ( IsNan(node_activation) )
			throw NumericException(STR_INVALID_ACTIVATION);

		AbstractSparseNode<NodeValue,WeightValue>::ToStream(s);

		_p_function->ToStream(s);

		return true;
	}

	template <class NodeValue, class WeightValue>
	SparseNode<NodeValue, WeightValue>* SparseNode<NodeValue,WeightValue>::Address(std::ptrdiff_t index)
	{
		return (this + index);
	}

	template<class NodeValue, class WeightValue>
	std::ptrdiff_t SparseNode<NodeValue,WeightValue>::Offset(AbstractSparseNode<NodeValue,WeightValue>* p_abstract_node) const
	{
		SparseNode<NodeValue,WeightValue>* p_node = dynamic_cast<SparseNode<NodeValue,WeightValue>*>(p_abstract_node);
		if (! p_node)
			throw SparseLibException(OFFSET_ERROR);

		return (p_node - this);
	}

	template <class NodeValue, class WeightValue>
	bool SparseNode<NodeValue, WeightValue>::FromStream(istream& s)
	{


		AbstractSparseNode<NodeValue,WeightValue>::FromStream(s);

		SquashingFunctionFactory factory;
		_p_function = factory.FromStream(s);


		return true;
	}





} // end of SparseImplementationLib

#endif // include guard

