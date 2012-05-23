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
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include "XMLRunParameter.h"
#include "MiindLibException.h"

using namespace MiindLib;

string XMLRunParameter::GetHandlerName(istream& s) const
{
	string dummy;
	s >> dummy;

	if (dummy != this->Tag() )
		throw MiindLibException("XMLRunParameter tag expected");

	s >> dummy;
	string ret;
	ret = this->UnWrapTag(dummy);
	return ret;
}

bool XMLRunParameter::GetCanvasValue(istream& s) const
{
	string dummy;
	s >> dummy;
	string bool_value;
	bool_value = this->UnWrapTag(dummy);
	if (bool_value != "TRUE" && bool_value != "FALSE")
		throw MiindLibException("Can't decode boolean value");
	return (bool_value == "TRUE") ? true : false;
}

bool XMLRunParameter::GetWriteState(istream& s) const
{
	string dummy;
	s >> dummy;
	string bool_value;
	bool_value = this->UnWrapTag(dummy);
	if (bool_value != "TRUE" && bool_value != "FALSE")
		throw MiindLibException("Can't decode boolean value");
	return (bool_value == "TRUE") ? true : false;
}

bool XMLRunParameter::GetWriteNet(istream& s) const
{
	string dummy;
	s >> dummy;
	string bool_value;
	bool_value = this->UnWrapTag(dummy);
	if (bool_value != "TRUE" && bool_value != "FALSE")
		throw MiindLibException("Can't decode boolean value");
	return (bool_value == "TRUE") ? true : false;
}

CanvasParameter XMLRunParameter::GetCanvasParameter(istream& s) const
{
	CanvasParameter par;
	par.FromStream(s);
	return par;
}

XMLRunParameter::XMLRunParameter(istream& s):
_handler_name(GetHandlerName(s)),
_b_canvas(GetCanvasValue(s)),
_b_file(GetWriteState(s)),
_b_write_net(GetWriteNet(s)),
_par_canvas(GetCanvasParameter(s))
{

	this->DecodeCanvasVector(s);

	string dummy;
	s >> dummy;
	if (dummy != this->ToEndTag(this->Tag()) )
		throw MiindLibException("Expected XMLRunParameter end tag");

}


XMLRunParameter::XMLRunParameter
(
	const string&			name,
	bool					b_canvas,
	bool					b_file,
	bool					b_write_net,
	const vector<string>&	vec_names,
	const CanvasParameter&	par_canvas
):_handler_name(name),
_b_canvas(b_canvas),
_b_file(b_file),
_b_write_net(b_write_net),
_par_canvas(par_canvas),
_vector_canvas_nodes(vec_names)
{}

XMLRunParameter::~XMLRunParameter()
{
}

bool XMLRunParameter::ToStream(ostream& s) const
{
	s << this->Tag()	<<"\n";
	s << this->WrapTag(_handler_name,"SimulationName")					<< "\n";
	s << this->WrapTag((_b_canvas? "TRUE": "FALSE"),"OnScreen")			<< "\n";
	s << this->WrapTag((_b_file? "TRUE": "FALSE"),"WithState")			<< "\n";
	s << this->WrapTag((_b_write_net? "TRUE" : "FALSE"), "WriteNet")	<< "\n";
	this->_par_canvas.ToStream(s);
	ostringstream str;
	BOOST_FOREACH( const string& name, _vector_canvas_nodes)
		str << "Name=\"" << name << "\" ";
	s << this->WrapTag(str.str().substr(0,str.str().size()-1),"CanvasNodes")<< "\n";
	s << this->ToEndTag(this->Tag())										<< "\n";

	return true;
}
bool XMLRunParameter::FromStream(istream&)
{
	return false;
}

string XMLRunParameter::Tag() const
{
	return "<SimulationIO>";
}

void XMLRunParameter::DecodeCanvasVector(istream& s)
{
	string dummy;
	getline(s,dummy);
	getline(s,dummy);
	dummy = this->UnWrapTag(dummy);
	
	boost::char_separator<char> sep("\"");
	boost::tokenizer<boost::char_separator<char> > tokens(dummy, sep);
	boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();
	for (	boost::tokenizer<boost::char_separator<char> >::const_iterator iter = tokens.begin(); iter != tokens.end(); iter++ ){
		string name = "";
		if (*iter != "Name=" ){
			_vector_canvas_nodes.push_back(*iter);
			iter++; // get rid of white space for the next "Name="
			if (iter == tokens.end() )
				break;
		}
	}
}