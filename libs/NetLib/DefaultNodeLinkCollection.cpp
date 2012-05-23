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


#include <algorithm>
#include "DefaultNodeLinkCollection.h"

using namespace NetLib;

DefaultNodeLinkCollection::DefaultNodeLinkCollection
(
		Number number_of_nodes,
		bool b_threshold
):
_b_is_valid(true),
_id_current_node_minus_one(0),
_number_of_nodes(number_of_nodes),
_b_threshold(b_threshold),
_vector_of_predecessors(InitializePredecessors())
{
}


DefaultNodeLinkCollection::~DefaultNodeLinkCollection()
{
}

NodeLink DefaultNodeLinkCollection::pop()
{
	vector<NodeId>::iterator iter_erase = _vector_of_predecessors.begin() + 
										  _id_current_node_minus_one;

	_vector_of_predecessors.erase(iter_erase);

	NodeId id(_id_current_node_minus_one + 1 );

	NodeLink link_return(id,_vector_of_predecessors);

	_vector_of_predecessors.insert(iter_erase,id);

	return link_return;
}




vector<NodeId> DefaultNodeLinkCollection::InitializePredecessors() const
{
	vector<NodeId> vector_return(_number_of_nodes);

	return vector_return;
}


Number DefaultNodeLinkCollection::NumberOfNodes () const
{
	throw 1;
}

Number DefaultNodeLinkCollection::NumberOfPredecessors ( NodeId id ) const
{
	throw 1;
}

DefaultNodeLinkCollection* DefaultNodeLinkCollection::Clone() const
{
	throw 1;
}

bool DefaultNodeLinkCollection::IsValid() const
{
	return _b_is_valid;
}

bool DefaultNodeLinkCollection::ToStream(ostream& s) const
{

	return true;
}

bool DefaultNodeLinkCollection::FromStream(istream&)
{
	return true;
}

string DefaultNodeLinkCollection::Tag() const
{
	return string("");
}

