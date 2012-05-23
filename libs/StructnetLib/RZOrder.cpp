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
#include "RZOrder.h"
#include "StructnetLibException.h"

using namespace StructnetLib;

RZOrder::RZOrder
(
	Index id,
	const vector<Pair>&		vec_pair,
	const vector<Layer>&	vec_desc
):
_id(id),
_vec_pair(vec_pair),
_vec_desc(vec_desc),
_imp(LayeredArchitecture(vec_desc)),
_id_pt(Map(0))
{
}

NodeId RZOrder::Id() const
{
	if ( _id_pt >= _vec_pair.size() )
		return  NodeId(-1);
	return _vec_pair[_id_pt].first;
}

const PhysicalPosition& RZOrder::Position() const
{
	if (_id_pt >= _vec_pair.size() )
		throw StructnetLibException("Try to call Id() beyond vector boundary");
	return _vec_pair[_id_pt].second;
}

bool StructnetLib::operator!=
(
	const RZOrder& lhs,
	const RZOrder& rhs
)
{
	return ( lhs._vec_pair != rhs._vec_pair || lhs._id != rhs._id);
}

Index RZOrder::Map(Index index) const
{
	// this should never happen
	assert(index <= _vec_pair.size());

	// if index points beyond the size of the _vec_pair, nothing should happen because
	// it is an end object
	if (index == _vec_pair.size())
		return static_cast<Index>(_vec_pair.size());

	// determine in which Layer this index falls if the index would point to a NodeId
	// in the feedforward network

	Number acc = 0;
	Layer l_current = 0;
	for (Layer layer = static_cast<Layer>(_vec_desc.size()) - 1; layer >=0; layer--)
	{
		acc += _imp.NumberOfNodesInLayer(layer);
		if ( index < acc )
		{
			l_current = layer;
			acc -= _imp.NumberOfNodesInLayer(layer);
			break;
		}
	}

	Index rank = index - acc;

	// Now reconstruct the  vector entry (= NodeId - 1) from the Layer and the rank
	Index  i = _imp.BeginId(l_current)._id_value + rank - 1;
	return i;
}

		
RZOrder& RZOrder::operator++()	
{ 	
	// prefix 
	++_id;
	_id_pt = this->Map(_id);
	return *this;
}

RZOrder RZOrder::operator+(int i)
{
	_id += i;
	_id_pt = this->Map(_id);
	return *this;
}