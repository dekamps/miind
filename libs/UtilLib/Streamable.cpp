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
//      If you use this software in work leading to a scientific publication, should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include <cassert>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include "ConcreteStreamable.h"
#include "Streamable.h"
#include "UtilLibException.h"

using namespace UtilLib;
using namespace std;

Streamable::Streamable():
Named("")
{
}

Streamable::Streamable(const Streamable& rhs):
Named(rhs)
{
}

Streamable::~Streamable()
{
}


string Streamable::ToEndTag(const string& tag) const
{
	string str_return = tag;
	str_return.insert(1,"/");

	return str_return;
}

string Streamable::DeTag(const string& tag) const {
	assert( tag.find('<') == 0);
	string str_ret = tag;
	str_ret.erase(str_ret.size()-1);
	str_ret.erase(str_ret.begin());
	return str_ret;
}

string Streamable::UnWrapTag(const string& s) const {

	boost::char_separator<char> sep("<>");
	boost::tokenizer<boost::char_separator<char> > tokens(s, sep);
	boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();

	if ( (*(++it))[0] == '/')
	{
		return string(""); // tag was empty
	}
	else
		return *it;
}

string Streamable::InsertNameInTag(const string& tag, const string& name) const
{

	boost::char_separator<char> sep("<>");
	boost::tokenizer<boost::char_separator<char> > tokens(tag, sep);
	boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();
	ostringstream s;
	s << "<" << *it << " Name=\"" << name << "\">"; 
	return s.str();
}

bool Streamable::StripNameFromTag(string* p_s, const string& dummy) const
{
	boost::char_separator<char> sep("\"");
	boost::tokenizer<boost::char_separator<char> > tokens(dummy, sep);
	boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();

	if (*it != " Name=") return false;
	it++;
	*p_s = *it;
	it++;

	return true;
}

string Streamable::AddAttributeToTag
(
	const string& tag,
	const string& attribute,
	const string& value
) const
{
	ConcreteStreamable streamable;
	AttributeList list;
	string tag_name = tag;
	
	if (! streamable.DecodeTag(&tag_name,&list) )
		throw UtilLibException("Could not decode tag");
	
	pair<string,string> p;
	p.first  = attribute;
	p.second = value;
	list._vec_attributes.push_back(p);

	return CompileTag(tag_name,list);
}

bool Streamable::DecodeTag
(
	string*			p_string,
	AttributeList*	p_list
)
{	
	string& tag = *p_string;

	string cpystr = tag;
	if (tag[0] != '<' || ( (*cpystr.rbegin() != '>') && (*(++cpystr.rbegin()) != '>')) ) 
		return false;

	boost::char_separator<char> sep("<>");
	boost::tokenizer<boost::char_separator<char> > tokens(tag, sep);
	boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();

	boost::char_separator<char> sepw(" \t");
	boost::tokenizer<boost::char_separator<char> > tokensw(*it, sepw);
	boost::tokenizer<boost::char_separator<char> >::iterator itw = tokensw.begin();

	*p_string = *itw;
	
	// Now a good separation of all tags is achieved. The only problem is that if attribute values themselves contain white spaces they must
	// be put together again
	vector<string> vec_pairs;
	string dummy= "";
	while(++itw != tokensw.end() ){
		if (*(itw->rbegin()) !=  '\"')
			dummy += *itw + " ";
		else
		{
			dummy += *itw;
			vec_pairs.push_back(dummy);
			dummy = "";
		}
	}
	
	BOOST_FOREACH(string s, vec_pairs)
	{
		boost::char_separator<char> sepeq("=");
		boost::tokenizer<boost::char_separator<char> > tokeneq(s, sepeq);
		boost::tokenizer<boost::char_separator<char> >::iterator iteq = tokeneq.begin();
		pair<string,string> p;
		p.first  = *iteq++;
		p.second = *iteq;
		p_list->_vec_attributes.push_back(p);
	}

	return true;
}

string Streamable::CompileTag(const string& tag, const AttributeList& list) const
{
	ostringstream str;
	str << "<" << tag;

	BOOST_FOREACH(const AttributePair& p,list._vec_attributes)
		str << " " << p.first << "=" << p.second;
	
	str << ">";

	return str.str();
}
