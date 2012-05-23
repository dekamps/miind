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
#include <cstdlib>
#include <boost/tokenizer.hpp>
#include "../UtilLib/UtilLib.h"
#include "NodeLinkCollection.h"
#include "NodeLinkCollectionPrivate.h"
#include "BasicDefinitions.h"
#include "NetLibException.h"


using namespace NetLib;
using namespace UtilLib;

NodeLinkCollection::NodeLinkCollection(istream& s):
_b_is_valid(true),
_vector_of_node_links(ParseInputStream(s))
{
}

NodeLinkCollection::NodeLinkCollection(const vector< NodeLink >& vec_link ):
_b_is_valid(true),
_vector_of_node_links(vec_link)
{
}


NodeLinkCollection::NodeLinkCollection(const NodeLinkCollection& rhs):
_b_is_valid(rhs._b_is_valid),
_vector_of_node_links(rhs._vector_of_node_links)
{
}


NodeLinkCollection::~NodeLinkCollection()
{
}


const NodeLink& NodeLinkCollection::operator [](size_t n_index) const
{
	return _vector_of_node_links[n_index];
}

Number NodeLinkCollection::NumberOfNodes() const
{
	return static_cast<Number>(_vector_of_node_links.size());
}
Number NodeLinkCollection::NumberOfPredecessors(NodeId nid) const
{
	int n_ind = static_cast<int>(_vector_of_node_links.size());
	assert( nid._id_value <= _vector_of_node_links[n_ind-1].MyNodeId()._id_value );
	assert( nid._id_value > 0 );

	while ( n_ind-- >  0 )
	{
		if ( _vector_of_node_links[n_ind].MyNodeId() == nid )
			return _vector_of_node_links[n_ind].Size();
		// else loop on
	}

	// code shouldn't reach this
	assert ( false );

	// close all return paths
	return 0;
}

NodeLink NodeLinkCollection::pop()
{

	// Ignore the slight inefficiency by this statement
	_b_is_valid = false;

	NodeLink link_ret = _vector_of_node_links[_vector_of_node_links.size()-1];
	_vector_of_node_links.pop_back();
	return link_ret;
}

bool NodeLinkCollection::IsValid() const
{
	return _b_is_valid;
}

NodeLinkCollection* NodeLinkCollection::Clone() const
{
	return new NodeLinkCollection(*this);
}

Number NodeLinkCollection::size() const
{
	return static_cast<Number>(_vector_of_node_links.size());
}

ostream& NetLib::operator<<
	(
		ostream& s, 
		const NodeLinkCollection& collection
	)
{
	collection.ToStream(s);
	return s;
}

bool NodeLinkCollection::ToStream(ostream& s) const
{
	s << Tag() <<"\n";
	
	for 
	(
		Index i = 0;
		i < NumberOfNodes();
		i++
	)
		s << _vector_of_node_links[i];

	s << ToEndTag(Tag()) << "\n";

	return true;
}

bool NodeLinkCollection::FromStream(istream& s)
{
	string str;

	s >> str;
	if ( str != Tag() )
		throw NetLibException("NodeLinkCollection tag expected");
	while (s)
	{
		s >> str;
	}
	return true;
}

string NodeLinkCollection::Tag() const
{
	return STRING_COLLECTION_HEADER;
}

std::vector<NodeLink> NodeLinkCollection::ParseInputStream(std::istream& s)
{
	 typedef boost::tokenizer<boost::char_separator<char> >  tokenizer;
 	vector<NodeLink> vec_parsed;
	string str;

	s >> str;
	if (str != STRING_COLLECTION_HEADER)
		throw NetLibException("NodeLinkCollection tag expected");

	int n = 0;
	while (s)
	{
		s >> str;

		boost::char_separator<char> sep("<>");
		tokenizer tokens(str, sep);
 
		tokenizer::iterator tok_iter = tokens.begin();
		int n_id = atoi((*tok_iter).c_str());

		if (n_id != ++n)
		{
			if ( str == ToEndTag(Tag()) )
				break;
			else
				throw NetLibException("NodeId list must be consecutive");
		}

		NodeId id(n_id);

		vector<NodeId> predec;
		boost::char_separator<char> sep_comma(",");
		if ( ++tok_iter != tokens.end() )
		{
			tokenizer token_comma(*(tok_iter),sep_comma);

			tokenizer::iterator tok_comma;

			for 
			(
				tok_comma =  token_comma.begin();
				tok_comma != token_comma.end();
				tok_comma++
			)
				predec.push_back(NodeId(atoi((*tok_comma).c_str())));
		}
		NodeLink link(id,predec);
		vec_parsed.push_back(link);
	}
	return vec_parsed;
}
