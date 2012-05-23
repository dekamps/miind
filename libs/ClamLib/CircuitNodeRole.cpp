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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifdef WIN32
#pragma warning(disable: 4267)
#pragma warning(disable: 4996)
#endif 

#include <cassert>
#include "CircuitNodeRole.h"

using namespace ClamLib;

CircuitNodeRole::CircuitNodeRole():
TNamed("",""),
_x(0.0),
_y(0.0),
_z(0.0),
_f(0.0),
_isOutput(false),
_isPositive(true),
_incoming(0)
{}

CircuitNodeRole::CircuitNodeRole
(
	const string& name, 
	UInt_t		type,
	Float_t		x,
	Float_t		y,
	Float_t		z,
	Float_t		f,
	bool isOutput,
	bool isPositive
):
TNamed(name.c_str(),""),
_type(type),
_x(x),
_y(y),
_z(z),
_f(f),
_isOutput(isOutput),
_isPositive(isPositive),
_incoming(0)
{
}

bool CircuitNodeRole::AddIncoming(IndexWeight weight)
{
	try {
		_incoming.push_back(weight);
	}
	catch (std::bad_alloc& )
	{
		return false;
	}
	return true;
}

UInt_t CircuitNodeRole::size() const
{
	return _incoming.size();
}

const IndexWeight& CircuitNodeRole::operator [] (UInt_t i) const
{
	assert(i < _incoming.size());
	return _incoming[i];
}

DynamicLib::SpatialPosition CircuitNodeRole::Position() const
{
	return DynamicLib::SpatialPosition(_x,_y,_z,_f);
}

const std::vector<IndexWeight>& CircuitNodeRole::IncomingVec() const
{
	return _incoming;
}

UInt_t CircuitNodeRole::Type() const
{
	return _type;
}

bool
CircuitNodeRole::isOutput() const
{
	return _isOutput;
}

bool
CircuitNodeRole::isPositive() const
{
	return _isPositive;
}

bool ClamLib::operator==(const CircuitNodeRole& role, const char* p)
{
	string str_role(role.GetName());
	string str_name(p);

	return (str_role == str_name);
}
