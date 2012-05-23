// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_ABSTRACTNODELINKCOLLECTION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_ABSTRACTNODELINKCOLLECTION_INCLUDE_GUARD

#include "../UtilLib/UtilLib.h"
#include "NodeId.h"
#include "NodeLink.h"

using UtilLib::Number;

namespace NetLib
{
	//! Abstract base class of NodeLinkCollections

	//! a NodeLinkCollection is essentially a stack of NodeLinks. A NodeLink described which nodes (characterised by
	//! their NodeId) input to a given Node. Hence a NodeLinkCollection provides a complete description of the network
	//! structure. In practice one sometimes wants to make information more explicit, for example, if the network
	//! has layed structure, whether it is feedforward, etc. Subclasses of AbstractNodeLinkCollection can provide
	//! such information.
	class AbstractNodeLinkCollection : public Streamable
	{
	public:

		//! destructor
		virtual ~AbstractNodeLinkCollection() = 0;

		//! total number of nodes in the network, for each node there is a link in the collection
		virtual Number NumberOfNodes() const = 0;

		virtual Number NumberOfPredecessors( NodeId ) const = 0;
		
		virtual NodeLink pop() = 0;

		virtual AbstractNodeLinkCollection* Clone() const = 0;

		virtual bool IsValid() const = 0;

		virtual string Tag () const;

		virtual bool ToStream(ostream&) const;

		virtual bool FromStream(istream&);
	protected:

		//! provide a total number of input nodes from a given Id
		Number ConnectionCount
		(
			NodeId id, 
			Number nr_nodes
		) const ;


	}; // end of AbstractNodeLinkCollection

} // end of NetLib

#endif // include guard
