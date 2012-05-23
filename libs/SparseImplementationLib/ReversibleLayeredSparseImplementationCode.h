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
#ifndef _CODE_LIBS_NETLIB_REVERSIBLELAYEREDSPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_REVERSIBLELAYEREDSPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "ReversibleLayeredSparseImplementation.h"

namespace SparseImplementationLib
{

	template <class ReversibleNode>
	ReversibleLayeredSparseImplementation<ReversibleNode>::ReversibleLayeredSparseImplementation(LayeredArchitecture* p_architecture):
	LayeredSparseImplementation<ReversibleNode>
		(
			p_architecture
		)
	{
	}

	template <class ReversibleNode>
	ReversibleLayeredSparseImplementation<ReversibleNode>::ReversibleLayeredSparseImplementation(istream& s):
	LayeredSparseImplementation<ReversibleNode>(s)
	{
	}

	template <class ReversibleNode>
	ReversibleLayeredSparseImplementation<ReversibleNode>::~ReversibleLayeredSparseImplementation()
	{
	}
		
	template <class ReversibleNode>
	ReversibleLayeredSparseImplementation<ReversibleNode>&
		ReversibleLayeredSparseImplementation<ReversibleNode>::operator=
	(
		const ReversibleLayeredSparseImplementation<ReversibleNode>& rhs
	)
	{
		if (this == &rhs)
			return *this;

		LayeredSparseImplementation<ReversibleNode>::operator=(rhs);
		return *this;
	}

	template <class ReversibleNode>
	bool ReversibleLayeredSparseImplementation<ReversibleNode>::ToStream(ostream& s) const
	{
		return LayeredSparseImplementation<ReversibleNode>::ToStream(s);
	}

	template <class ReversibleNode>
	double ReversibleLayeredSparseImplementation<ReversibleNode>::ReverseInnerProduct(NodeId id) const 
	{
		assert( id._id_value >= 0 && id._id_value < static_cast<int>(this->NumberOfNodes()) );
		return _vector_of_nodes[id._id_value].ReverseInnerProduct();
	}

	template <class ReversibleNode>
	double ReversibleLayeredSparseImplementation<ReversibleNode>::ReverseThresholdInnerProduct
	(
		NodeId id_start_next_layer, 
		NodeId id_end_next_layer
	) const 
	{
		return _vector_of_nodes[0].ReverseThresholdInnerProduct(id_start_next_layer, id_end_next_layer);
	}

} // end of SparseImplementation

#endif // include guard
