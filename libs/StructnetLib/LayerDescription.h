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

#ifndef _CODE_LIBS_STRUCNET_LAYERDESCRIPTION_INCLUDE_GUARD
#define _CODE_LIBS_STRUCNET_LAYERDESCRIPTION_INCLUDE_GUARD


#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"

using std::ostream;
using std::istream;
using UtilLib::Number;
using NumtoolsLib::Rational;


namespace StructnetLib
{

	//! LayerDescription

	struct LayerDescription
	{

		Number _nr_x_pixels;				// nr. neurons in x
		Number _nr_y_pixels;				// nr. neurons in y
		Number _nr_features;				// nr. of 'feature' or 'orientation' layers
		Number _size_receptive_field_x;	    // size of receptive field in x for this layer
		Number _size_receptive_field_y;		// size of receptive field in y for this layer
		Rational _nr_x_skips;					// nr. of 'skips' in x : must NOT be zero
		Rational _nr_y_skips;					// nr. of 'skips' in y : must NOT be zero

	}; // end of LayerDescription

	// comparison for use in vectors
	bool operator< (const LayerDescription&, const LayerDescription&);
	bool operator==(const LayerDescription&, const LayerDescription&);
	bool operator!=(const LayerDescription&, const LayerDescription&);

	// I/O of DescData
	istream& operator>>(istream&, LayerDescription&);
	ostream& operator<<(ostream&, const LayerDescription&);

// LayerStructure is meant to indicate the physical position of a neuron,
// it is usually associated with a NeuronId

} // end of Strucnet

#endif // include guard
