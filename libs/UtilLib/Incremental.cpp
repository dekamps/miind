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
#ifdef WIN32
#pragma warning(disable: 4996)
#endif

#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "Incremental.h"
#include "UtilLibException.h"

using namespace boost;
using namespace UtilLib;
using namespace std;

namespace {
	const string text_exception("Incremental serialization corrupted");
	const int INDEX_NAME  = 0;
	const int INDEX_START = 1;
	const int INDEX_END   = 2;
	const int INDEX_INC   = 3;
	const int OPEN_TAG    = 1;
	const int OPEN_INC    = 3;
	const int STRING_TAG  = 5;
	const int CLOSE_INC   = 7;
	const int CLOSE_TAG   = 9;

}

Incremental::Incremental(istream& s):
Sequence(s)
{
	this->FromStream(s);
}

bool Incremental::FromStream(istream& s){
	string str;

	std::getline(s,str);
	vector<string> split_vec;
	split(split_vec,str,is_any_of("<>"));

	if (split_vec[OPEN_TAG]  != string("Sequence"))
		throw UtilLibException(text_exception);

	if (split_vec[OPEN_INC]  != string("Incremental") )
		throw UtilLibException(text_exception);

	if (split_vec[CLOSE_INC] != string("/Incremental") )
		throw UtilLibException(text_exception);

	if (split_vec[CLOSE_TAG] != string("/Sequence"))
		throw UtilLibException(text_exception);

	vector<string> split_values;
	split(split_values,split_vec[STRING_TAG],is_any_of(" "));

	_name = split_values[INDEX_NAME];

	try{
		_start	= lexical_cast<double>(split_values[INDEX_START]);
		_end	= lexical_cast<double>(split_values[INDEX_END]);
		_step	= lexical_cast<double>(split_values[INDEX_INC]);
	}
	catch(bad_lexical_cast&){

			throw UtilLibException(text_exception);
	}

	_nr_steps = static_cast<int>((_end - _start)/_step) + 1;
	return true;
}

bool Incremental::ToStream(ostream&) const
{
	return true;
}

string Incremental::Tag() const {
	return string("<Incremental>");
}

Incremental* Incremental::Clone() const
{
	return new Incremental(*this);
}

double Incremental::operator [](Index ind) const
{
	assert(ind < _nr_steps);
	return _start + _step*ind;
}

Index Incremental::BeginIndex() const
{
	return 0;
}

Index Incremental::EndIndex() const
{
	return _nr_steps;
}

string Incremental::Name() const
{
	return _name;
}
