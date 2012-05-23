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
//      If you use this software in work leading to a scientific publication, it would be cool if you would include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include "AbstractSquashingParameter.h"
#include <cassert>
using namespace std;
using namespace NetLib;

AbstractSquashingParameter::~AbstractSquashingParameter()
{
}

bool AbstractSquashingParameter::InsertParameter(Index index, double f_value)
{
	assert(index < _vector_of_parameters.size());
	_vector_of_parameters[index] = f_value;

	return true;
}

double AbstractSquashingParameter::GetParameter(Index index) const
{
	assert(index < _vector_of_parameters.size());
	return _vector_of_parameters[index];
}

ostream& NetLib::operator<<(ostream& s, const AbstractSquashingParameter& parameter)
{
	parameter.ToStream(s);
	return s;
}

istream& NetLib::operator>>(istream& s, AbstractSquashingParameter& parameter)
{
	parameter.FromStream(s);
	return s;
}

