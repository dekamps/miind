
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NETLIB_DEFAULTLAYERNODELINKCOLLECTION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_DEFAULTLAYERNODELINKCOLLECTION_INCLUDE_GUARD

#include "AbstractNodeLinkCollection.h"
#include "BasicDefinitions.h"

namespace NetLib
{
	//! In a DefaultNodeLinkCollection, a feedforward architecture is defined with fully connected layers.

	//! This class is not used stand alone but invoked by a LayeredArchitecture. Thus it is possible to
	//! initialize a LayeredNetwork by a LayeredArchtecture, which in turn has been initialized by a vector<Layer>
	//! only

	class DefaultLayerNodeLinkCollection : public AbstractNodeLinkCollection
	{
	public:

		//! Default constructor, takes a description of the feedforward structure
		DefaultLayerNodeLinkCollection
		(
				const vector<Layer>&
		);

		//! copy constructor
		DefaultLayerNodeLinkCollection
		(
				const DefaultLayerNodeLinkCollection&
		);

		//! virtual destructor
		virtual ~DefaultLayerNodeLinkCollection();

		//! total number of nodes (there is a link for each node)
		virtual Number		NumberOfNodes		() const;


		virtual Number		NumberOfPredecessors	(NodeId) const;

		//! clone operator 
		virtual DefaultLayerNodeLinkCollection* Clone() const;

		virtual bool IsValid() const;


		virtual NodeLink	pop();

	private:

		vector<NodeLink> InitializeNodeLinks
		(
			const vector<Layer>&
		);


		bool				_b_is_valid;
		Number				_number_of_nodes;
		vector <NodeLink>	_vec_link;

	};

} //end of NetLib

#endif // include guard
