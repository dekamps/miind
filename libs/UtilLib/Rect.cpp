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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include "Rect.h"

using namespace UtilLib;

Rect::Rect( const RectAttributes& ra ):
_attrect(ra)
{
}

Rect::~Rect()
{
}


void Rect::GeneratePs
		(
			PositionInCm PosCm, 
			SizeInCm SizeCm, 
			Angle angle, 
			ostream& s
			) const
{
	// set the correct color
	s << _attrect._color._red   << " " 
	  << _attrect._color._green << " "
	  << _attrect._color._blue  << " setrgbcolor\n";

	// translate to the position

	s << PosCm._f_x << " cm " << PosCm._f_y << " cm  translate\n";

	// rotate frame

	s << angle << " rotate\n";

	// draw rectangle

	s << " 0 " << " " << " 0 " << " \n";
	s << _attrect._width << " cm " << _attrect._length << " cm  rectfill\n";

	// rotate back
	s << -angle << " rotate\n";

	// translate back to origin

	s << -PosCm._f_x << " cm " << -PosCm._f_y << " cm  translate\n";
}

