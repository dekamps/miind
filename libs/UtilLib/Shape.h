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
#ifndef _CODE_LIBS_UTIL_SHAPE_INCLUDE_GUARD
#define _CODE_LIBS_UTIL_SHAPE_INCLUDE_GUARD


#include <iostream>
#include "PositionInCm.h"
#include "SizeInCm.h"

using std::ostream;

namespace UtilLib
{
	//! Some useful typedefs

	typedef float Angle;
	typedef float Length;
	typedef float Width;

	//! Color enum
	enum Color       { RED, GREEN, BLUE, YELLOW, WHITE, BLACK };

	//! Shape

	//! Abstract base class for shapes that are to be put on a Page

	class Shape {
	public:

		// a shape generates its own PS contribution to a page:
		// it gets size, position and the stream to write to from the page

		//! Every shape should generate its own PS 

		virtual void GeneratePs
						(
							PositionInCm,
							SizeInCm,
							Angle,
							ostream&
						) const = 0; 

		//! virtual destructor
		virtual	     ~Shape    () = 0;

	}; // end of Shape


	inline Shape::~Shape()
	{
	}

} // end of Util

#endif // include guard
