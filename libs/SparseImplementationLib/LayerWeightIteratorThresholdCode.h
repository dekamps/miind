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
#ifndef _CODE_LIBS_NETLIB_LAYERITERATORCODE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LAYERITERATORCODE_INCLUDE_GUARD


// Copyright (c) 2005 - 2007 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, it would be cool if you would include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "LayerWeightIteratorThreshold.h"

namespace SparseImplementationLib
{
	template <class NodeType>
	LayerWeightIteratorThreshold<NodeType>::LayerWeightIteratorThreshold():
	_p_first_node_of_layer(0),
	_p_node_threshold(0),
	_index_iterator(0)
	{
	}

	template <class NodeType>
	LayerWeightIteratorThreshold<NodeType>::LayerWeightIteratorThreshold
	(
		NodeType*  p_node_threshold,
		NodeType*  p_node_first,
		int    index_iterator,
		NodeId id_first_next_layer,
		NodeId id_last_next_layer
	):
	_p_first_node_of_layer(p_node_first),
	_p_node_threshold(p_node_threshold),
	_index_iterator(index_iterator),
	_first_id_next_layer(id_first_next_layer),
	_last_id_next_layer(id_last_next_layer)
	{
	}

	template <class NodeType>
	LayerWeightIteratorThreshold<NodeType>& 
		LayerWeightIteratorThreshold<NodeType>::operator++()
	{
		++_index_iterator;
		return *this;
	}

	template <class NodeType>
	LayerWeightIteratorThreshold<NodeType>
		LayerWeightIteratorThreshold<NodeType>::operator++(int)
		{
			LayerWeightIteratorThreshold iter = *this;
			++*this;
			return iter;
		}

	template <class NodeType>
	NodeType*
		LayerWeightIteratorThreshold<NodeType>::operator->() const
		{
			return (_index_iterator == 0 ) ?
                   _p_node_threshold : 
			       _p_first_node_of_layer + _index_iterator - 1;
				   
		}


	template <class NodeType>
	double LayerWeightIteratorThreshold<NodeType>::ReverseInnerProduct() const
		{
			if (_index_iterator > 0)
				return operator->()->ReverseInnerProduct();
			else 
				// Check if we don't try to do a ReverseInnerProduct on an output layer
			  if ( _first_id_next_layer == NetLib::THRESHOLD_ID ||
			       _last_id_next_layer  == NetLib::THRESHOLD_ID    )
					 throw SparseLibIteratorException(string("Attempt to calculate a reverse inner product on an output layer"));
				else
					return operator->()->ReverseThresholdInnerProduct
					(
						_first_id_next_layer,
						_last_id_next_layer
					);
		}

} // end of SparseImplementationLib


#endif // include guard
