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

#include "ActivityRect.h"

using namespace UtilLib;
using namespace StructnetLib;

ColorTable GLCOLOR;

ActivityRect::ActivityRect( const F_Matrix& mat, Color Max, Color Min ):
_atmat(mat),
_col_max(Max),
_col_min(Min)
{
}

void ActivityRect::DrawFrame( std::ostream& s, SizeInCm Size ) const
{
	// set line color
	s << _atrect._colout._red    << " " 
	  << _atrect._colout._green  << " " 
	  << _atrect._colout._blue   << " setrgbcolor\n";


	// calculate horizontal line width

	float horwidth = _atrect._outthick*Size._f_x_size;

	// set line width

	s << horwidth << " cm  setlinewidth\n";


	s << " 0  0  moveto\n";
	s <<  Size._f_x_size << " cm  0 rlineto\n";
	s << "0 "           <<  Size._f_y_size << " cm rlineto\n";
	s << -Size._f_x_size << " cm  0 rlineto\n";
	s << "closepath\n";
	s << "stroke\n";
}

void ActivityRect::DrawGrid( std::ostream& s, SizeInCm Size) const
{
	// set the inner line width
	float width = _atrect._inthick*Size._f_x_size;
	s <<  width                 << " cm setlinewidth\n";
	s << _atrect._colin._red    << " "
	  << _atrect._colin._green  << " "
	  << _atrect._colin._blue   << " setrgbcolor\n";

	size_t nr_x_lines = _atmat.NrXdim() - 1;
	size_t nr_y_lines = _atmat.NrYdim() - 1;

	float xdist = Size._f_x_size/(nr_x_lines + 1);
	float ydist = Size._f_y_size/(nr_y_lines + 1);

	for ( size_t n_x_lines = 0; n_x_lines < nr_x_lines; n_x_lines++ )
	{
		s << xdist*(n_x_lines + 1)		<< " cm 0 moveto\n";
		s << " 0 "                      << Size._f_y_size << " cm  rlineto\n";
	}
	for ( size_t n_y_lines = 0; n_y_lines < nr_y_lines; n_y_lines++ )
	{
		s << " 0 cm " <<  ydist*(n_y_lines + 1) << " cm moveto\n";
		s << Size._f_x_size << " cm 0 cm rlineto\n";
	}
	s << "stroke\n";
}


void ActivityRect::DrawMatrix( std::ostream& s, SizeInCm Size) const
{
	// ( rotations taken care off in mother routine )
	// How many elements in x and y are there ?

	size_t n_x_rectangles = _atmat.NrXdim();
	size_t n_y_rectangles = _atmat.NrYdim();

	// What are the x and y sizes of each element ?

	float rect_x_size = Size._f_x_size/n_x_rectangles;
	float rect_y_size = Size._f_y_size/n_y_rectangles;


	s << -rect_x_size << " cm 0 translate\n";


	for ( size_t n_x_rect = 0; n_x_rect < n_x_rectangles; n_x_rect++ )
	{
		s << rect_x_size << " cm 0 translate\n";
		s << "0 " << rect_y_size*n_y_rectangles << " cm translate\n";

		// loop over x	
		// loop over y
			
		for ( size_t n_y_rect = 0 ; n_y_rect < n_y_rectangles; n_y_rect++ )
		{
			// translate one square down
			s << " 0 " << -rect_y_size << " cm translate\n"; 

			// set the correct color
			
			RGBValue color_element = ConvertColor
										(
											_atmat
											(
												static_cast<long>(n_x_rect),
												static_cast<long>(n_y_rect)
											)
										);

			s << color_element._red		<< " "
			  << color_element._green	<< " "
			  << color_element._blue    << " setrgbcolor\n";

			// draw the rectangle

			s << " 0  0 " << rect_x_size << " cm " << rect_y_size << " cm rectfill\n";
		}
	}

	s << -rect_x_size*(n_x_rectangles - 1) << " cm 0 translate\n";	
}

RGBValue ActivityRect::ConvertColor( float x ) const
{
	assert ( x <= 1.0 && x >= -1.0 );

	if ( _col_max != WHITE || _col_min != BLACK )

	// Make a color graph, where White or Black can be one of the chosen colors
	{
		if ( x > 0 )
			return  x*GLCOLOR[_col_max];
		else
			return -x*GLCOLOR[_col_min];
	}
	
	else
	
	// Make a Black&White graph

	{

		float colval = 0.5F*(1 + x);
		return RGBValue(colval,colval,colval);

	}
}

void ActivityRect::GeneratePs
					(
						PositionInCm	Position, 
						SizeInCm		Size, 
						Angle			angle, 
						ostream& s
					)
const
{
	// translate to lower left corner
	s << Position._f_x << " cm " << Position._f_y << " cm translate\n";
	// rotate frame

	s << angle << " rotate\n";

	// Loop over all matrix elements and draw them

	DrawMatrix(s,Size);

	// Draw the inner lines

	DrawGrid(s,Size);	

	// Draw the outer frame

	DrawFrame(s,Size);
	
	// rotate back	

	s << -angle << " rotate\n";

	// translate back to origin

	s << -Position._f_x << " cm " << Position._f_y << " cm translate\n";
}
