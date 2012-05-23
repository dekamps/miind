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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_CLAMLIB_TOROOTLAYERDESCRIPTION_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_TOROOTLAYERDESCRIPTION_INCLUDE_GUARD

#include "../StructnetLib/StructnetLib.h"
#include "RootLayerDescription.h"
#include "ClamLibException.h"

using StructnetLib::LayerDescription;

namespace ClamLib {

	inline RootLayerDescription ToRootLayerDescription
	(
		const LayerDescription& desc
	)
	{
		RootLayerDescription desc_ret;

		desc_ret._nr_features = desc._nr_features;
		desc_ret._nr_x_pixels = desc._nr_x_pixels;
		desc_ret._nr_y_pixels = desc._nr_y_pixels;

		desc_ret._size_receptive_field_x = desc._size_receptive_field_x;
		desc_ret._size_receptive_field_y = desc._size_receptive_field_y;

		// not prepared to handle rational skip sizes in ClamLib
		if (
				desc._nr_x_skips.getRemainder() != 0 ||
				desc._nr_y_skips.getRemainder() != 0
			)
			throw ClamLibException("Rational skip size is not acceptable");

		desc_ret._nr_x_skips  = desc._nr_x_skips.getValue();
		desc_ret._nr_y_skips  = desc._nr_y_skips.getValue();

		return desc_ret;
	}
}

#endif // include guard
