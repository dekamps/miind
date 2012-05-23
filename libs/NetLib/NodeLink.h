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
#ifndef _CODE_LIBS_NETLIB_NODELINK_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_NODELINK_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <vector>        
#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "NodeId.h"

using std::vector;
using std::ostream;
using UtilLib::Index;
using UtilLib::Number;


namespace NetLib
{
	//! NodeLink
	//! Author: Marc de Kamps
	//! Date:   14-07-1999
	//! Short description: A NodeLink is a NodeId and a list of related NodeID's.
	//! The related Nodes may be input or output Nodes, depending on the way that
	//! a Network will be defined		

	class NodeLink 
	{
	public:

		//! default constructor
		NodeLink();

		//!  a NodeLink is typically a NodeId, and a list of NodeId's which are the predecessors of this node
		NodeLink(NodeId, const vector<NodeId>&);
				
		//! number of predecessors 
		Number	Size () const;

		//! NodeId where this link relates to, the Node of which the predecessors are to be specified.
		NodeId	MyNodeId() const;

		//! acces the list of predecessors
		NodeId	operator[](Index) const;

		//! add a NeuronId to the list of related neurons
		void	PushBack(NodeId);

		friend ostream& operator<<(ostream&, const NodeLink&);

	private:

		NodeId			_id_of_this_link;		  // NodeID of this link
		vector<NodeId>	_vector_of_related_nodes; // list of related Nodes

	};
		
	ostream& operator<<(ostream&, const NodeLink& );

} // end of NetLib

#endif // include guard
