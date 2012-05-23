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

#include "../NetLib/NetLib.h"
#include "LayerDescription.h"

using namespace StructnetLib;
using namespace NetLib;

bool StructnetLib::operator< ( const LayerDescription&, const LayerDescription& )
{
	throw UnimplementedException(string("This function is unimplemented"));
}

bool StructnetLib::operator==( const LayerDescription& d1, const LayerDescription& d2 )
{

	if ( d1._nr_x_pixels            != d2._nr_x_pixels            || 
		 d1._nr_y_pixels            != d2._nr_y_pixels            || 
		 d1._nr_features            != d2._nr_features            ||
		 d1._size_receptive_field_x != d2._size_receptive_field_x || 
		 d1._size_receptive_field_y != d2._size_receptive_field_y || 
		 d1._nr_x_skips				!= d2._nr_x_skips			  ||
		 d1._nr_y_skips				!= d2._nr_y_skips)
		return false;
	else
		return true;
}

bool StructnetLib::operator!=(const LayerDescription& d1, const LayerDescription& d2)
{
	return ( d1 == d2 ) ? false : true;
}


ostream& StructnetLib::operator<<(ostream& s, const LayerDescription& dd)
{
	s << dd._nr_x_pixels << "\t"   << dd._nr_y_pixels << "\t" << dd._nr_features << "\t" 
	  << dd._size_receptive_field_x << "\t" << dd._size_receptive_field_y << "\t" 
	  << dd._nr_x_skips << "\t"  << dd._nr_y_skips << "\n";
	return s;
}

std::istream& StructnetLib::operator>>(istream& s, LayerDescription& dd )
{
	s >> dd._nr_x_pixels   >> dd._nr_y_pixels   >> dd._nr_features   >> 
		 dd._size_receptive_field_x >> dd._size_receptive_field_y >> dd._nr_x_skips >> 
		 dd._nr_y_skips;
	return s;
}
