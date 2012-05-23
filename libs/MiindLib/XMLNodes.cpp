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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include <boost/lexical_cast.hpp>
#include "MiindLibException.h"
#include "XMLNodes.h"

using namespace MiindLib;

XMLNode::XMLNode
(
	const string&	type,
	const string&	name,
	const string&	alg
):
_type_name(type),
_alg_name(alg)
{
	this->SetName(name);
}

XMLNode::XMLNode(istream& s):
Persistant(s)
{
	string dummy;
	getline(s,dummy);

	AttributeList list;
	bool b_res = this->DecodeTag(&dummy,&list);
	if (! b_res)
		throw MiindLibException("Decoding tag failed");

	//  presumably this is down to reaching the end of the node list:
	if (list._vec_attributes.size() == 0){
		this->SetValidity(false);
		this->SetOffendingString(dummy);
	}
	else {
		if (dummy != "Node")
			throw MiindLibException("Expected Node tag");
		this->SetName(list["name"]);
		this->_type_name = list["type"];
		this->_alg_name  = list["algorithm"];
		this->SetValidity(true);
	}
}

XMLNode::XMLNode(const XMLNode& rhs):
Persistant(rhs),
_type_name(rhs._type_name),
_alg_name(rhs._alg_name)
{
}

XMLNode::~XMLNode()
{
}

bool XMLNode::ToStream(ostream& s) const
{
	string tag = this->Tag();
	tag = this->AddAttributeToTag(tag,"name",      "\"" + this->GetName() + "\"");
	tag = this->AddAttributeToTag(tag,"type",      "\"" + _type_name      + "\"");
	tag = this->AddAttributeToTag(tag,"algorithm", "\"" + _alg_name       + "\"");
	s << tag;
	s << this->ToEndTag(this->Tag()) <<"\n";

	return true;
}

bool XMLNode::FromStream(istream& s)
{
	return false;
}

string XMLNode::Tag() const
{
	return "<Node>";
}

DynamicLib::NodeType MiindLib::FromValueToType(const string& type_name){

	if ( type_name == "EXCITATORY")
		return DynamicLib::EXCITATORY;

	if ( type_name == "EXCITATORY_BURST" )
		return DynamicLib::EXCITATORY_BURST;

	if (type_name == "INHIBITORY" )
		return DynamicLib::INHIBITORY;

	if (type_name == "INHIBITORY_BURST" )
		return DynamicLib::INHIBITORY_BURST;

	if (type_name == "NEUTRAL")
		return DynamicLib::NEUTRAL;

	throw MiindLibException("Unknown NodeType in tag");
}
