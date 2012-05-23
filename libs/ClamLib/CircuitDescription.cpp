// Copyright (c) 2005 - 2010 Dave Harrison, Marc de Kamps
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
#pragma warning(disable:4267)
#pragma warning(disable:4996)
#endif
#include <cassert>
#include <boost/foreach.hpp>
#include "CircuitDescription.h"
#include "ClamLibException.h"

using namespace boost;
using namespace ClamLib;

CircuitDescription::CircuitDescription(UInt_t nr_nodes):
_b_is_setup(0),
_nr_nodes(nr_nodes),
_index(0),
_vec_role(nr_nodes),
_vec_io(0),
_name_external(""),
_index_external(0)
{
}

UInt_t CircuitDescription::push_back(const CircuitNodeRole& role)
{
	assert(_index < _vec_role.size());
	if (_b_is_setup)
		throw ClamLibException("Adding to a completed circuit is prohibited");

	_vec_role[_index++] = role;
	if ( _index == _nr_nodes)
		FinalizeDescription();

	return _index-1; 
}

const CircuitNodeRole& CircuitDescription::operator[](UInt_t i) const
{
	assert (i < _vec_role.size());
	return _vec_role[i];
}

UInt_t CircuitDescription::size() const
{
	return _vec_role.size();
}

void  CircuitDescription::push_back_io(const InputOutputPair& pair)
{
	_vec_io.push_back(pair);
}

const std::vector<CircuitNodeRole>&
	CircuitDescription::RoleVec() const
{
	return _vec_role;
}

const std::vector<InputOutputPair>&
	CircuitDescription::IOVec() const
{
	return _vec_io;
}

void CircuitDescription::FinalizeDescription()
{
	_b_is_setup = true;

	// insert all indices
	BOOST_FOREACH(CircuitNodeRole& role, _vec_role){
		BOOST_FOREACH(IndexWeight& iw, role._incoming)
		{
			vector<CircuitNodeRole>::iterator iter;
			iter = std::find(_vec_role.begin(),_vec_role.end(),iw._name_predecessor);
			if (iter != _vec_role.end() )
				iw._index = iter - _vec_role.begin();
			else
				throw ClamLibException("Unknown name used as input");
		}
	}

	// don't forget the external
	vector<CircuitNodeRole>::iterator iter_external;
	iter_external = std::find(_vec_role.begin(),_vec_role.end(),_name_external);
	if (iter_external != _vec_role.end() )
		_index_external = iter_external - _vec_role.begin();
	else
		throw ClamLibException("Unknown name used for external");
}

void CircuitDescription::AddExternal(const char * p)
{
	_name_external = TString(p);
}

UInt_t CircuitDescription::IndexExternal() const
{
	return _index_external;
}

UInt_t CircuitDescription::IndexInCircuitByName(const char* p) const
{
	vector<CircuitNodeRole>::const_iterator iter;
	iter = std::find(_vec_role.begin(),_vec_role.end(),p);

	if (iter == _vec_role.end() )
		throw ClamLibException("Unknown circuit name");


	return (iter - _vec_role.begin());
}
