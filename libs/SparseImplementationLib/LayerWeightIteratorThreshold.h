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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_LAYERWEIGHTITERATORTHRESHOLD_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_LAYERWEIGHTITERATORTHRESHOLD_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

// Name:   LayerIterator
// Author: Marc de Kamps
// Date:   04-08-2003
// Short description: Is produced by a LayeredImplementation. Iterates over all Nodes in a Layer
//                     

#include <vector>
#include "BasicDefinitions.h"
#include "LayerWeightIterator.h"
#include "SparseLibIteratorException.h"

using std::vector;
using NetLib::Layer;
using NetLib::NodeId;


namespace SparseImplementationLib
{

	template <class NodeType>
	class LayerWeightIteratorThreshold
	{
	public:

		LayerWeightIteratorThreshold();
		LayerWeightIteratorThreshold
		(
			NodeType*,
			NodeType*,
			int,
			NodeId,
			NodeId
		);
	
		LayerWeightIteratorThreshold& operator++();
		LayerWeightIteratorThreshold  operator++(int);
		
		NodeType* operator->() const;

		double ReverseInnerProduct() const;

		template <class NodeType_> 
			friend bool operator!=
					(
						const LayerWeightIteratorThreshold<NodeType_>&,
						const LayerWeightIteratorThreshold<NodeType_>&
					);

	private:

		NodeType*       _p_first_node_of_layer;
		NodeType*       _p_node_threshold;
		int		_index_iterator;
		NodeId          _first_id_next_layer;
		NodeId          _last_id_next_layer;

	}; // end of LayerWeightIteratorThreshold

	template <class NodeType>
	inline bool operator!=
	(
		const LayerWeightIteratorThreshold<NodeType>& iter_left,
		const LayerWeightIteratorThreshold<NodeType>& iter_right
	)
	{ 

		return 
		(
			iter_left._index_iterator  != 
			iter_right._index_iterator   
		);
	}

} // end of SparseImplementationLib


#endif // include guard
