
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
#ifndef _CODE_LIBS_NETLIB_LAYEREDARCHITECTURE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LAYEREDARCHITECTURE_INCLUDE_GUARD

#include <vector>
#include "AbstractArchitecture.h"
#include "BasicDefinitions.h"

using std::vector;

namespace NetLib
{

  //! LayeredArchitecture
  class LayeredArchitecture : public AbstractArchitecture
    {
    public:

        //! default ctor
        LayeredArchitecture();

		//! Feedforward network, completely connected
		LayeredArchitecture
		(
			const vector<Layer>&,
			bool b_threshold = false
		);

		//! Feedforward network, structure defined by NodeLinkCollection
		LayeredArchitecture
		(
			const vector<Layer>&,
			AbstractNodeLinkCollection*,
			bool b_threshold = false
		);

		//! copy ctor
		LayeredArchitecture
		(
			const LayeredArchitecture&
		);

		virtual ~LayeredArchitecture();

		//! Number of Layers
		Number NumberOfLayers() const;

		//! The starting Id of each layer
		vector<NodeId> BeginIdVector() const;

		//! Give the number of links between layer n and layer n+1
		vector<Number> NumberConnectionsVector() const;

		//! vector description of layers
		vector<Layer>  LayerVector() const;

		//! input nodes are the first layer
		virtual Number NumberOfInputNodes() const;

		//! output nodes are the last layer
		virtual Number NumberOfOutputNodes() const;

	private:

	Number	NumberInLayers		
		(
		 const vector<Layer>&
		) const;


	vector<NodeId>  InitializeBeginIds
			(
			 const vector<Layer>&
			) const;

	vector<Number>  InitializeNumberFrom
			(
			) const;

		bool		_b_external_collection;
		Number		_number_of_nodes;
		vector<Layer>	_vector_of_layers;
		vector<NodeId>  _vector_of_begin_ids;
		vector<Number>	_vector_number_from;

	}; // end of Layered Architecture

} // end of NetLib

#endif // include guard
