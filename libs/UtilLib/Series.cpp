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

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "Series.h"
#include "UtilLibException.h"

using namespace boost;
using namespace UtilLib;

namespace {

	const int OPEN_TAG     = 1;
	const int OPEN_SERIES  = 3;
	const int STRING_TAG   = 5;
	const int CLOSE_SERIES = 7;
	const int CLOSE_TAG    = 9;

}

Series::Series(istream& s):
Sequence(s),
_name(string("")),
_vec_series(0)
{
	this->FromStream(s);
}

bool Series::FromStream(istream& s)
{
	string str;

	std::getline(s,str);
	vector<string> split_vec;
	split(split_vec,str,is_any_of("<>"));

	if (split_vec[OPEN_TAG] != string("Sequence") )
		throw UtilLibException(STR_SERIES_CORRUPTED);

	if (split_vec[OPEN_SERIES] != string("Series") )
		throw UtilLibException(STR_SERIES_CORRUPTED);

	if (split_vec[CLOSE_SERIES] != string("/Series") )
		throw UtilLibException(STR_SERIES_CORRUPTED);

	if (split_vec[CLOSE_TAG] != string("/Sequence") )
		throw UtilLibException("Series serialization corrupted");
	
	vector<string> split_values;
	split(split_values,split_vec[STRING_TAG],is_any_of(" "));
	int i = 0;
	BOOST_FOREACH(string str_val,split_values)
	{
		if (i++ == 0)
			_name = str_val;
		else
		{
			try {
				_vec_series.push_back(lexical_cast<double>(str_val));
			}
			catch(bad_lexical_cast& ){
				throw UtilLibException("Series serialization corrupted");
			}
		}
	}	
	return true;
}

bool Series::ToStream(ostream& s) const
{
	return true;
}

string Series::Tag() const
{
	return string("<Series>");
}

Index Series::BeginIndex() const
{
	return 0;
}

Index Series::EndIndex() const
{
	return static_cast<Index>(_vec_series.size());
}

double Series::operator [](Index ind) const
{
	assert (ind < _vec_series.size());
	return _vec_series[ind];
}

Series* Series::Clone() const
{
	return new Series(*this);
}

string Series::Name() const
{
	return _name;
}
