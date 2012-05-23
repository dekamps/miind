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

#include "Page.h"
#include "DinA.h"
#include "PaperFormatError.h"

using namespace UtilLib;


Page::Page( PaperFormat Format ):
_current_origin(0,0)
{
	switch (Format)
	{
		case A4:
			_paper_size = DinA(4);
			break;
		case A5:
			_paper_size = DinA(5);
			break;
		default:
			throw PaperFormatError();
			break;
	}
}

void Page::ShapeOnPage(Shape* p_sh, PositionInCm pos, SizeInCm siz, Angle angle)
{
	_vec_shapes_on_page.push_back(p_sh);
	_vec_pos.push_back(pos);
	_vec_siz.push_back(siz);
	_vec_angle.push_back(angle);
}

void Page::ToStream(std::ostream& s) const
{
	s << "%!PS\n";
	s << "/cm { 28.35 mul } def\n";

	for ( size_t index = 0; index < _vec_pos.size(); index++ )
		_vec_shapes_on_page[index]->GeneratePs(_vec_pos[index],
											   _vec_siz[index],
											   _vec_angle[index],
											   s);

	s << "showpage\n";
} 
