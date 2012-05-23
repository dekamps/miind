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
#ifndef _CODE_LIBS_STRUCNET_ACTIVITYRECT_INCLUDE_GUARD
#define _CODE_LIBS_STRUCNET_ACTIVITYRECT_INCLUDE_GUARD

#include "ActivityRectangleAttributes.h"
#include "../UtilLib/UtilLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"

using NumtoolsLib::F_Matrix;
using UtilLib::Angle;
using UtilLib::Color;
using UtilLib::RED;
using UtilLib::GREEN;
using UtilLib::PositionInCm;
using UtilLib::Shape;
using UtilLib::SizeInCm;

namespace StructnetLib
{
	//! ActivityRect

	//! rectangle showing activities

	class ActivityRect : public Shape 
	{
	public:

		//! Postive activities are red by default, negative are green
		ActivityRect
			(
				const F_Matrix&, 
				Color = RED, 
				Color = GREEN
			);

		//! virtual destructor
		virtual		~ActivityRect(){}

		//! generate Postscript on the output stream

		virtual void GeneratePs
						(
							PositionInCm,
							SizeInCm,
							Angle,
							ostream&
						) const;
	

		void		 ConvertColor( Color, Color );
		Color		 CurrentMaxColor() const;
		Color		 CurrentMinColor() const;

	private:

		void		DrawFrame	(ostream&, SizeInCm) const;
		void		DrawMatrix	(ostream&, SizeInCm) const;
		void		DrawGrid	(ostream&, SizeInCm) const;
		RGBValue	ConvertColor(float) const;

		ActivityRectangleAttributes	_atrect;		// color and size attribute definitions
		F_Matrix					_atmat;			// local copy of the matrix to display
		Color						_col_max;		// color for the positive scale values
		Color						_col_min;		// color for the negative scale values

	}; // end of ActivityRect

} // end of Strucnet

#endif // include guard
