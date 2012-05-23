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
#ifndef _CODE_LIBS_NETLIB_LAYERSTRUCTURE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LAYERSTRUCTURE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include <cmath>
#include <string>
#include "Architecture.h"

#include "LayerWeightIterator.h"
#include "ReverseLayerWeightIterator.h"

using std::string;
using std::istream;

// Created:				27-09-1999
// Author:				Marc de Kamps

namespace NetLib
{
  //! LayerStructure

  class LayerStructure 
    {
    public:

      //! Read LayerStructure from stream
      LayerStructure(istream&);

      
      LayerStructure(const Architecture&);
      LayerStructure(const LayerStructure&);
      ~LayerStructure();

		LayerStructure&	operator=(const LayerStructure&);

		// network properties:

		size_t	NumberOfNodes      () const;        // number of Nodes, specified by Architecture
		size_t	NumberOfInputNodes () const;        // number of input Nodes
		size_t	NumberOfOutputNodes() const;        // number of output Nodes

		size_t	NumberOfLayers() const;             // number of Layers
		size_t	NumberOfNodesInLayer(Layer) const;  // number of Nodes in a given layer

		size_t  NrConnections	        () const;
		size_t  MaxNumberOfNodesInLayer () const;
		size_t  NrConnectionFrom(size_t) const;
		size_t  NrFstConInLayer (size_t) const;

		NodeId  BeginId			(size_t) const;

		vector<size_t> ArchVec	()		 const;

		friend ostream& operator<<(ostream&, const LayerStructure&);

	private: 
	
		vector<size_t>   _vec_arch;
		vector<size_t>   _vec_connections_from;
		vector<NodeId>   _vec_begin_id;
		vector<size_t>   _vec_begin_connection;

		size_t _number_of_nodes;
		size_t _number_of_input_nodes;
		size_t _number_of_output_nodes;
		size_t _number_of_connections;

	}; // end of LayerStructure

} // end of NetLib

#endif // include guard
