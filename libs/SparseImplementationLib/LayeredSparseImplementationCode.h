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
#ifndef _CODE_LIBS_NETLIB_LAYEREDSPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LAYEREDSPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif


#include "LayeredSparseImplementation.h"

namespace SparseImplementationLib
{


	template <class NodeType>
	LayeredSparseImplementation<NodeType>::LayeredSparseImplementation(LayeredArchitecture* p_architecture):
	SparseImplementation<NodeType>(p_architecture),
	_layered_implementation(*p_architecture)
	{
	}

	template <class NodeType>
	LayeredSparseImplementation<NodeType>::LayeredSparseImplementation
	(
		const LayeredSparseImplementation<NodeType>& rhs
	):
	SparseImplementation<NodeType>(rhs),
	_layered_implementation(rhs._layered_implementation)
	{
	}

	template <class NodeType>
	LayeredSparseImplementation<NodeType>::LayeredSparseImplementation(istream& s):
	SparseImplementation<NodeType>(RemoveHeader(s)),
	_layered_implementation(s)
	{
		RemoveFooter(s);
	}

	template <class NodeType>
	LayeredSparseImplementation<NodeType>::~LayeredSparseImplementation()
	{
	}

	template <class NodeType>
	LayeredSparseImplementation<NodeType>&
			LayeredSparseImplementation<NodeType>::operator=
			(
				const LayeredSparseImplementation<NodeType>& rhs
			)
	{
		if (&rhs == this)
			return *this;
		else
		{
			// ensure correct transfer of base class
			SparseImplementation<NodeType>::operator=(rhs);

			// member transfer
			_layered_implementation = rhs._layered_implementation;
	
			return *this;
		}
	}

	template <class NodeType>
	vector<Layer> LayeredSparseImplementation<NodeType>::ReverseLayerDescription() const
	{
		vector<Layer> vector_return(this->NumberOfLayers());
	
		for (size_t n_index = 0; n_index < this->NumberOfLayers(); n_index++)
			vector_return[n_index] = NumberOfNodesInLayer(this->NumberOfLayers() - 1 - n_index);

		return vector_return;
	}

	template <class NodeType>
	istream& LayeredSparseImplementation<NodeType>::RemoveHeader(istream& s) const
	{			
		string str;
		s >> str; // Absorb file header
	
		if (str != LayeredSparseImplementation<NodeType>::Tag() )
			throw NetworkParsingException(string("Expected LayeredSparseImplementation header"));
	
		return s;
	}

	template <class NodeType>
	istream& LayeredSparseImplementation<NodeType>::RemoveFooter(istream& s) const
	{			
		string str;
		s >> str; // Absorb file header
	
		if (str != LayeredSparseImplementation<NodeType>::ToEndTag(Tag()) )
			throw NetworkParsingException(string("Expected LayeredSparseImplementation footer"));

		return s;
	}
	
	template <class NodeType>
	Number LayeredSparseImplementation<NodeType>::NumberOfLayers() const
	{
		return _layered_implementation.NumberOfLayers();
	}

	template <class NodeType>
	NodeId LayeredSparseImplementation<NodeType>::BeginId(Layer layer) const
	{
			return _layered_implementation.BeginId(layer);
	}

	template <class NodeType>
	NodeId LayeredSparseImplementation<NodeType>::EndId(Layer layer) const
	{
			return _layered_implementation.EndId(layer);
	}

	template <class NodeType>
	Number LayeredSparseImplementation<NodeType>::NumberOfNodesInLayer(Layer layer) const
	{
			return _layered_implementation.NumberOfNodesInLayer(layer);
	}

	template <class NodeType>
	bool LayeredSparseImplementation<NodeType>::ToStream(ostream& s) const
	{
		s << LayeredSparseImplementation<NodeType>::Tag() << "\n";

		// Write base classes to stream

		if ( ! SparseImplementation<NodeType>::ToStream(s) )
			return false;

		_layered_implementation.ToStream(s);

		s << this->ToEndTag(LayeredSparseImplementation<NodeType>::Tag()) << "\n";

		return true;
	}

	template <class NodeType>
	string LayeredSparseImplementation<NodeType>::Tag() const
	{
		return STR_LAYEREDSPARSEIMPLEMENTATION_HEADER;
	}

	template <class NodeType>
	bool LayeredSparseImplementation<NodeType>::FromStream
	(
		istream& s
	)
	{
		return true;
	}

	template <class NodeType>
	LayerWeightIterator<NodeType> 
		LayeredSparseImplementation<NodeType>::begin
		(
			Layer n_layer, 
			LayerWeightIterator<NodeType>* p_dummy
	) 
	{
		return LayerWeightIterator<NodeType>
				(
					&_vector_of_nodes[this->BeginId(n_layer)._id_value]
				);
	}


	template <class NodeType>
		LayerWeightIterator<NodeType> LayeredSparseImplementation<NodeType>::end
		(
			Layer n_layer, 
			LayerWeightIterator<NodeType>* p_dummy
		)
	{
		NodeType* p_end_node = &_vector_of_nodes[0] + BeginId(n_layer)._id_value + NumberOfNodesInLayer(n_layer);
		return LayerWeightIterator<NodeType>(p_end_node);
	}


	template <class NodeType> 
		LayerWeightIteratorThreshold<NodeType> LayeredSparseImplementation<NodeType>::begin
		(
			Layer n_layer, 
			LayerWeightIteratorThreshold<NodeType>* p_dummy
		)
	{
		return LayerWeightIteratorThreshold<NodeType>
			(
				&_vector_of_nodes[0],
				&_vector_of_nodes[BeginId(n_layer)._id_value],
				0,
				( BeginIdNextHighestLayer(n_layer) ),
				( EndIdNextHighestLayer(n_layer) ) 
			);
	}

	template <class NodeType>
	LayerWeightIteratorThreshold<NodeType> 
		LayeredSparseImplementation<NodeType>::end
		(
			Layer n_layer, 
			LayerWeightIteratorThreshold<NodeType>* p_dummy
		) 
	{


		return LayerWeightIteratorThreshold<NodeType>
			(
				&_vector_of_nodes[0],
				&_vector_of_nodes[BeginId(n_layer)._id_value],
				NumberOfNodesInLayer(n_layer) + 1,
				BeginIdNextHighestLayer(n_layer),
				EndIdNextHighestLayer(n_layer)
			);
		
	}

	template <class NodeType>
	NodeId LayeredSparseImplementation<NodeType>::BeginIdNextHighestLayer(Layer n_layer) const
	{
		return (n_layer + 1 < NumberOfLayers() ) ?  BeginId(n_layer+1) : NodeId(0); 
	}


	template <class NodeType>
	NodeId LayeredSparseImplementation<NodeType>::EndIdNextHighestLayer(Layer n_layer) const
	{
		return (n_layer + 1 < NumberOfLayers() ) ?  NodeId( BeginId(n_layer+1)._id_value + NumberOfNodesInLayer(n_layer+1) - 1) : NodeId(0);
	}

} // end of SparseImplementationLib


#endif // include guard
