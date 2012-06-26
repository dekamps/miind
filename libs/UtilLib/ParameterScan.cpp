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
#include "ParameterScan.h"
#include "SequenceFactory.h"
#include "UtilLibException.h"

using namespace UtilLib;

ParameterScan::ParameterScan(istream& s):
_iter(BuildIter(s))
{
}

ParameterScan::~ParameterScan()
{
}

SequenceIteratorIterator ParameterScan::BuildIter(istream& s)
{
	SequenceIteratorIterator iteriter(true);

	string str;
	std::getline(s,str);

	if (str != this->Tag() )
		throw UtilLibException("ParameterScan object could not be constructed");
	// from now on build Sequences

	SequenceFactory fact;
	while (Sequence* p = fact.Create(s))
	{
		boost::shared_ptr<Sequence> p_sh(p);
		iteriter.AddLoop(*p_sh);
	}

	return iteriter;
}

string ParameterScan::Tag() const
{
	return string("<ParameterScan>");
}

bool ParameterScan::ToStream(ostream&) const
{
	return true;
}

bool ParameterScan::FromStream(istream&)
{
	return true;
}

SequenceIteratorIterator ParameterScan::begin() const
{
	return _iter;
}

SequenceIteratorIterator ParameterScan::end() const
{
	return _iter + _iter.size();
}
