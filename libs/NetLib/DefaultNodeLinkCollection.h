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
#ifndef _CODE_LIBS_NETLIB_DEFAULTNODELINKCOLLECTION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_DEFAULTNODELINKCOLLECTION_INCLUDE_GUARD

#include "AbstractNodeLinkCollection.h"
#include "NodeLink.h"

namespace NetLib
{

	class DefaultNodeLinkCollection : public AbstractNodeLinkCollection
	{
	public:

		//! Link every node to every other node
		DefaultNodeLinkCollection
		(
			Number, 
			bool 
			b_threshold = false
		);

		virtual ~DefaultNodeLinkCollection();

		//! allow an architecture to pop the next NodeLink from the collection
		virtual NodeLink	pop						();

		//! number of nodes in the collection
		virtual Number		NumberOfNodes			() const;

		//! number of predecessors of a given node
		virtual Number		NumberOfPredecessors	(NodeId) const;

		//! virtual construction
		virtual DefaultNodeLinkCollection* Clone() const;

		virtual bool IsValid() const;

		//! streaming tag
		virtual string Tag() const;

		//! output streaming
		virtual bool ToStream(ostream&) const;

		//! input streaming
		virtual bool FromStream(istream&);


	private:

		bool	_b_is_valid;
		int		_id_current_node_minus_one;
		Number	_number_of_nodes;
		bool    _b_threshold;
		
		vector<NodeId> InitializePredecessors() const;

		// precompute a vector that contains almost all predecessors
		vector<NodeId> _vector_of_predecessors;

	}; // end of DefaultNodeLinkCollection

} // end of NetLib

#endif // include guard 
