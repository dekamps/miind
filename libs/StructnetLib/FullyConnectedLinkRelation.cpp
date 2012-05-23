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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "../NetLib/NetLib.h"
#include "FullyConnectedLinkRelation.h"
#include "DescriptionException.h"
#include "InconsistentStructureException.h"
#include "LayerDescription.h"

#ifdef WIN32
#pragma warning( disable: 4786 )
#endif

using namespace NetLib;
using namespace StructnetLib;
using namespace UtilLib;

FullyConnectedLinkRelation::FullyConnectedLinkRelation
(
	Number n_layers,
	Number n_neurons_per_layer
)
{
	LayerDescription desc;
	desc._nr_x_pixels  = n_neurons_per_layer;
	desc._nr_y_pixels  = 1;
	desc._nr_x_skips   = n_neurons_per_layer;
	desc._nr_y_skips   = n_neurons_per_layer;
	desc._nr_features  = 1;
	desc._size_receptive_field_x = n_neurons_per_layer;
	desc._size_receptive_field_y = n_neurons_per_layer;


	for ( Index i = 0; i < n_layers; i++)
		_vector_layer_description.push_back(desc);
		
}

FullyConnectedLinkRelation::~FullyConnectedLinkRelation()
{
}

bool FullyConnectedLinkRelation::operator ()( const PhysicalPosition& In, const PhysicalPosition& Out ) const
{
	if ( Out._position_z == 0  ||  (Out._position_z - In._position_z != 1) )
		return false;

	return true;
}

const std::vector<LayerDescription>& FullyConnectedLinkRelation::VectorLayerDescription() const
{
	return _vector_layer_description;
}


vector<PhysicalPosition> FullyConnectedLinkRelation::VecLayerStruct() const
{
	vector<PhysicalPosition> vec_ret;

	for (Layer n_ind = 0; n_ind < _vector_layer_description.size(); n_ind++ )
	{
		PhysicalPosition current_layer;
		current_layer._position_x     = _vector_layer_description[n_ind]._nr_x_pixels;
		current_layer._position_y     = _vector_layer_description[n_ind]._nr_y_pixels;
		current_layer._position_z     = n_ind;
		current_layer._position_depth = _vector_layer_description[n_ind]._nr_features;

		vec_ret.push_back( current_layer);
	}

	return vec_ret;
}
