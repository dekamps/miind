// Copyright (c) 2005 - 2009 Marc de Kamps
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

#include "ReverseDenseOverlapLinkRelation.h"

using namespace std;
using namespace StructnetLib;

ReverseDenseOverlapLinkRelation::~ReverseDenseOverlapLinkRelation()
{
}

ReverseDenseOverlapLinkRelation::ReverseDenseOverlapLinkRelation
( 
	const SpatialConnectionistNet& net 
):
_p_net(&net)
{
    assert(_p_net->Dimensions().size() > 0);

	size_t n_max = _p_net->Dimensions().size() - 1;

	for (int n_ind = static_cast<int>(n_max); n_ind >= 0; n_ind-- )
	{
		LayerDescription current_data = _p_net->Dimensions()[n_ind];
		_vec_desc_data.push_back(current_data);
	}
}

bool ReverseDenseOverlapLinkRelation::operator()
	( 
		const PhysicalPosition& In,
		const PhysicalPosition& Out
	) const
{

	assert(_p_net->NumberOfLayers() > 0);

	// in order to compare. the z-values must be reversed !!
	Index n_max = _p_net->NumberOfLayers() - 1;

	PhysicalPosition In_forward  = In;
	PhysicalPosition Out_forward = Out;

	In_forward._position_z  = n_max - In_forward._position_z; 
	Out_forward._position_z = n_max - Out_forward._position_z;

	NodeId id_in  = _p_net->Id(In_forward);
	NodeId id_out = _p_net->Id(Out_forward);
	
	// reverse the input/output relationship of the forward net
	if ( _p_net->IsInputNeuronFrom(id_out,id_in) )
		return true;
	else
		return false;	
}

const vector<LayerDescription>&
	ReverseDenseOverlapLinkRelation::VectorLayerDescription() const
{
	return _vec_desc_data;
}
