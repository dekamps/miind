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
#ifndef _CODE_LIBS_STRUCNET_XORLINKRELATION_INCLUDE_GUARD
#define _CODE_LIBS_STRUCNET_XORLINKRELATION_INCLUDE_GUARD

#include "DenseOverlapLinkRelation.h"

namespace StructnetLib {

	//!A LinkRelation that implements and XOR network
	//!
	//!This XOR network is a simple test class for the Backpropagation algorithm

	class XORLinkRelation : public AbstractLinkRelation
	{
	public:

		XORLinkRelation():_vec_desc(VecDesc()){}

		virtual ~XORLinkRelation(){}

		virtual bool operator ()
		( 
			const PhysicalPosition& In, 
			const PhysicalPosition& Out 
		) const;


		virtual const vector<LayerDescription>& VectorLayerDescription() const;


	private:

		vector<LayerDescription> VecDesc() const;

		vector<LayerDescription> _vec_desc;

	};


	inline bool XORLinkRelation::operator ()
	(
		const PhysicalPosition& In,
		const PhysicalPosition& Out
	) const
	{
		if (	In._position_x == 0  && In._position_y == 0  && In._position_z == 0 && In._position_depth == 0  &&
			Out._position_x == 0 && Out._position_y == 0 && Out._position_z == 1 && Out._position_depth == 0 )
			return true;

		if (	In._position_x == 1  && In._position_y == 0  && In._position_z == 0 && In._position_depth == 0  &&
			Out._position_x == 0 && Out._position_y == 0 && Out._position_z == 1 && Out._position_depth == 0 )
			return true;


		if (	In._position_x == 0  && In._position_y == 0  && In._position_z == 0 && In._position_depth == 0  &&
			Out._position_x == 1 && Out._position_y == 0 && Out._position_z == 1 && Out._position_depth == 0 )
			return true;

		if (	In._position_x == 1  && In._position_y == 0  && In._position_z == 0 && In._position_depth == 0  &&
			Out._position_x == 1 && Out._position_y == 0 && Out._position_z == 1 && Out._position_depth == 0 )
			return true;


		if (	In._position_x == 0  && In._position_y == 0  && In._position_z == 1 && In._position_depth == 0  &&
			Out._position_x == 0 && Out._position_y == 0 && Out._position_z == 2 && Out._position_depth == 0 )
			return true;

		if (	In._position_x == 1  && In._position_y == 0  && In._position_z == 1 && In._position_depth == 0  &&
			Out._position_x == 0 && Out._position_y == 0 && Out._position_z == 2 && Out._position_depth == 0 )
			return true;

		return false;
	}


	inline const vector<LayerDescription>& XORLinkRelation::VectorLayerDescription() const 
	{
		return _vec_desc;
	}

	inline vector<LayerDescription> XORLinkRelation::VecDesc() const
	{
		vector<LayerDescription> vec_ret;

		LayerDescription desc;

		desc._nr_features = 1;
		desc._nr_x_pixels = 2;
		desc._nr_y_pixels = 1;
		desc._nr_x_skips  = 1;
		desc._nr_y_skips  = 1;
		desc._size_receptive_field_x = 1;
		desc._size_receptive_field_y = 1;
	
		vec_ret.push_back(desc);

		desc._nr_features = 1;
		desc._nr_x_pixels = 2;
		desc._nr_y_pixels = 1;
		desc._nr_x_skips  = 1;
		desc._nr_y_skips  = 1;
		desc._size_receptive_field_x = 2;
		desc._size_receptive_field_y = 1;

		vec_ret.push_back(desc);

		desc._nr_features = 1;
		desc._nr_x_pixels = 1;
		desc._nr_y_pixels = 1;
		desc._nr_x_skips  = 1;
		desc._nr_y_skips  = 1;
		desc._size_receptive_field_x = 1;
		desc._size_receptive_field_y = 1;

		vec_ret.push_back(desc);

		return vec_ret;
	}
}
#endif // include guard
