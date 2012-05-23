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
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include "PlotParameter.h"
#include "PopulistException.h"

using namespace PopulistLib;

PlotParameter::PlotParameter():
_id(NodeId(1)),
_t_begin(0.0),
_t_end(1.0)
{
}

PlotParameter::~PlotParameter()
{
}

string PlotParameter::Tag() const {
	return string("<PlotParameter>");
}

bool PlotParameter::ToStream(ostream& s) const
{
	s << Tag() << "\n";
	s << "<NodeId>" << _id		<< "</NodeId>\n";
	s << "<TBegin>" << _t_begin	<< "</TBegin>\n";
	s << "<TEnd>"	<< _t_end	<< "</TEnd>\n";
	s << this->ToEndTag(this->Tag());
	return true;
}

bool PlotParameter::FromStream(istream& s)
{
	string dummy;
	s >> dummy;
	if (dummy != this->Tag() )
		throw PopulistException("PlotParameter tag expected");

	s >> dummy;
	typedef boost::tokenizer<boost::char_separator<char> > 
		tokenizer;
	boost::char_separator<char> sep("<>");
	tokenizer tokens(dummy, sep);
	tokenizer::iterator tok_iter = tokens.begin();
	cout << *tok_iter << endl;
	if (*tok_iter != string("NodeId") )
		throw PopulistException("PlotParameter tag mismatch");
	Index ind = boost::lexical_cast<Index>(*(++tok_iter));
	_id = NodeId(ind);
	if (*(++tok_iter) != string("/NodeId") )
		throw PopulistException("PlotParameter tag mismatch");

	s >> dummy;
	tokenizer tokentb(dummy,sep);
	tok_iter = tokentb.begin();
	if (*tok_iter != string("TBegin") )
		throw PopulistException("PlotParameter tag mismatch");
	_t_begin = boost::lexical_cast<Time>(*(++tok_iter));

	s >> dummy;
	tokenizer tokente(dummy,sep);
	tok_iter = tokente.begin();
	if (*tok_iter != string("TEnd") )
		throw PopulistException("PlotParameter tag mismatch");
	_t_end = boost::lexical_cast<Time>(*(++tok_iter));

	s >> dummy;
	if (dummy != this->ToEndTag(this->Tag()) )
		throw PopulistException("Unexpected end tag for PlotPar");

	return true;
}

ostream& PopulistLib::operator<<(ostream& s, const PlotParameter& par){
	par.ToStream(s);
	return s;
}

istream& PopulistLib::operator>>(istream& s, PlotParameter& par){
	par.FromStream(s);
	return s;
}