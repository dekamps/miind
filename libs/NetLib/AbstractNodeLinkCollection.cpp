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

#include <cassert>
#include "AbstractNodeLinkCollection.h"
#include "LocalDefinitions.h"

using namespace NetLib;

AbstractNodeLinkCollection::~AbstractNodeLinkCollection()
{
}

string AbstractNodeLinkCollection::Tag() const
{
	return STR_ABSTRACTNODELINKCOLLECTION_TAG;
}

bool AbstractNodeLinkCollection::ToStream(ostream&) const
{
	return true;
}

bool AbstractNodeLinkCollection::FromStream(std::istream &)
{
	return true;
}

Number AbstractNodeLinkCollection::ConnectionCount(NodeId id, Number nr_neurons) const
{
	// Calculates the number of input Nodes from all Nodes, starting
	// at id, and ending at  id + NodeId(nr_neurons)
	// Typical use is to calculate the number of Links between two
	// layers of Nodes in a sparse network

	int ncount = 0;

	// This assert fires for instance when somebody has added
	// an element with NodeId 0 to a NodeLinkCollection.
	// This Id is reserved for the threshold Id and is inserted
	// automatically:
	assert (id._id_value > 0);

	// Obvious check
	assert ( static_cast<size_t>(id._id_value) <= NumberOfNodes() );

	NodeId current_id;
	for (int n_ind = 0; n_ind < static_cast<int>(nr_neurons); n_ind++ )
	{
		current_id = NodeId(id._id_value + n_ind);
		ncount += this->NumberOfPredecessors(current_id);
	}

	return static_cast<Number>(ncount);
}