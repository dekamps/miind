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
#ifdef WIN32
#pragma warning(disable: 4996)
#endif

#include <sstream>
#include <cassert>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include "Series.h"
#include "Sequence.h"
#include "SequenceFactory.h"
#include "Incremental.h"
#include "UtilLibException.h"

using namespace UtilLib;
using namespace std;

namespace {
	const int OPEN_TAG = 3;
}

SequenceFactory::SequenceFactory()
{
}

Sequence* SequenceFactory::Create(istream& s)
{
	// a sequence should be represented on a single line
	string line, to_base_class;
	getline(s,line);
	to_base_class = line;
	istringstream istr(line);

	vector<string> split_vec;
	boost::split(split_vec,line,boost::is_any_of("<>"));

	// create any series and increment object
	boost::shared_ptr<Sequence> p_series	= boost::shared_ptr<Series>(new Series);
	boost::shared_ptr<Sequence> p_increment	= boost::shared_ptr<Incremental>(new Incremental);

	Sequence* p_ret;
	if (split_vec.size() - 1 < static_cast<Index>(OPEN_TAG) )
		return NULL;

	if ( split_vec[OPEN_TAG] == p_series->DeTag(p_series->Tag()) ){
		p_ret = new Series(istr);
		return p_ret;
	}

	if ( split_vec[OPEN_TAG] == p_increment->DeTag(p_increment->Tag()) ){
		p_ret = new Incremental(istr);
		return p_ret;
	}

	return NULL;
}