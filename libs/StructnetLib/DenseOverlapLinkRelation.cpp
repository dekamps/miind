// Copyright (c) 2005 - 2007 Marc de Kamps
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
#include "DenseOverlapLinkRelation.h"
#include "DescriptionException.h"
#include "InconsistentStructureException.h"

#ifdef WIN32
#pragma warning( disable: 4786 )
#endif

using namespace NetLib;
using namespace StructnetLib;
using namespace UtilLib;

DenseOverlapLinkRelation::DenseOverlapLinkRelation()
{
}

DenseOverlapLinkRelation::DenseOverlapLinkRelation( std::istream& s )
{

	string str;
	s >> str;
	if ( str != "<denseoverlap>" )
		throw DescriptionException(string("Dense overlapped header failed"));


	LayerDescription field;

	while (s)
	{
		s >> field._nr_x_pixels>> field._nr_y_pixels >> field._nr_features >> field._size_receptive_field_x >> 
			field._size_receptive_field_y >> field._nr_x_skips >> field._nr_y_skips;
		if (s.good())
			_vector_layer_description.push_back(field);	
		else 
			break;
	}
}

DenseOverlapLinkRelation::DenseOverlapLinkRelation(const vector<LayerDescription>& vector_layer_description):
_vector_layer_description(vector_layer_description)
{
}

DenseOverlapLinkRelation::~DenseOverlapLinkRelation()
{
}

bool DenseOverlapLinkRelation::operator ()( const PhysicalPosition& In, const PhysicalPosition& Out ) const
{
	if ( Consistent() == false )
		return false;

	if ( Out._position_z == 0  ||  (Out._position_z - In._position_z != 1) )
		return false;

	// deal with x
	// the minimal x that is in the receptive field is given by:
	// changed by Korbo for Rational skips
	Index x_min = (Index)(Out._position_x*_vector_layer_description[Out._position_z]._nr_x_skips).getValue();
	// the maximimum x is determined by the receptive field
	Index x_max = x_min + _vector_layer_description[Out._position_z]._size_receptive_field_x - 1;

	if ( In._position_x < x_min || In._position_x > x_max )
		return false;

	// deal with y
	// the minimal x that is in the receptive field is given by:
	// changed by Korbo for Rational skips
	Index y_min = (Index)(Out._position_y*_vector_layer_description[Out._position_z]._nr_y_skips).getValue();
	// the maximimum x is determined by the receptive field
	Index y_max = y_min + _vector_layer_description[Out._position_z]._size_receptive_field_y - 1;

	if ( In._position_y < y_min || In._position_y > y_max )
		return false;

	// still here ? then there is a connection

	return true;
}

const std::vector<LayerDescription>& DenseOverlapLinkRelation::VectorLayerDescription() const
{
	return _vector_layer_description;
}

bool DenseOverlapLinkRelation::ToStream( std::ostream& out) const {

	out << "<denseoverlap>" << endl;

	for (Index n_ind = 0; n_ind < static_cast<Index>(_vector_layer_description.size()); n_ind++)
		out << _vector_layer_description[n_ind]._nr_x_pixels << "\t" << _vector_layer_description[n_ind]._nr_y_pixels << "\t" << _vector_layer_description[n_ind]._nr_features << "\t\t" <<
			_vector_layer_description[n_ind]._size_receptive_field_x << "\t\t" << _vector_layer_description[n_ind]._size_receptive_field_y << "\t\t" << _vector_layer_description[n_ind]._nr_x_skips <<
			"\t\t" << _vector_layer_description[n_ind]._nr_y_skips << std::endl;

	out << "</denseoverlap>" << endl;

	return true;
}


bool  DenseOverlapLinkRelation::Consistent() const 
{
	// layer 0 is always ok
	size_t n_ind;
	for ( n_ind = 1; n_ind < _vector_layer_description.size(); n_ind++ )
	{	
		// Check on the number of skips. This must always be greater than zero,
		// to prevent division by zero in a consistency check of the network.
		// If the number of skips does not matter for a layer, as sometimes happens
		// in an output layer, set it to 1.

		if (_vector_layer_description[n_ind]._nr_x_skips*_vector_layer_description[n_ind]._nr_y_skips == 0)
			throw InconsistentStructureException(string("Inconsistent Layer Structure"));

		// Check if there are no dangling input nodes for this layer
		
		int nr_x_skips = _vector_layer_description[n_ind]._nr_x_skips.getValue();
		int nr_y_skips = _vector_layer_description[n_ind]._nr_y_skips.getValue();

		int nr_previous_x = _vector_layer_description[n_ind-1]._nr_x_pixels;
		int nr_previous_y = _vector_layer_description[n_ind-1]._nr_y_pixels;

		int nr_receptive_x = _vector_layer_description[n_ind]._size_receptive_field_x;
		int nr_receptive_y = _vector_layer_description[n_ind]._size_receptive_field_y;

		int nr_this_x = _vector_layer_description[n_ind]._nr_x_pixels;
		int nr_this_y = _vector_layer_description[n_ind]._nr_y_pixels;

		int number_rest_x =    nr_previous_x 
			                   -nr_receptive_x 
							   -nr_x_skips*nr_this_x; 

		int number_rest_y =    nr_previous_y
			                   -nr_receptive_y 
							   -nr_y_skips*nr_this_y;
		
		if ( ( number_rest_x != -1 ) ||
			 ( number_rest_y != -1 ) )
			 return false;
	}
	return true;
}

vector<PhysicalPosition> DenseOverlapLinkRelation::VecLayerStruct() const
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

