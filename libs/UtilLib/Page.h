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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_UTIL_PAGE_INCLUDE_GUARD
#define _CODE_LIBS_UTIL_PAGE_INCLUDE_GUARD

#include <vector>
#include <iostream>
#include "Shape.h"

using std::ostream;
using std::vector;

namespace UtilLib
{

	//! Useful enums

	enum PaperFormat { A0, A1, A2, A3, A4, A5 };

	//! Page

	//! Objectholder. Calls upon each object to produce PS on a real printer page

	class  Page {
	public:

		// create a page of given format:
		// a page holds shapes and their absolute positions (in cm)
		// and sizes, as the shape also can give its own size, relative positions
		// can be calculated if necessary

		//! Create printerPage with format

		Page( PaperFormat = A4 ); 

		//! add a shape
		void  ShapeOnPage
				(
					Shape*, 
					PositionInCm, 
					SizeInCm, 
					Angle
				); 
		//! generate Postscript on a stream
		void  ToStream(ostream&) const;					

		//! give page size
		SizeInCm PageSizeInCm()	 const;					

	private:

		PositionInCm				_current_origin;
		SizeInCm					_paper_size;

		vector< const Shape* >	_vec_shapes_on_page;
		vector< PositionInCm >	_vec_pos;
		vector< SizeInCm >		_vec_siz;
		vector< Angle >			_vec_angle;

	}; // end of Page

} // end of Util

#endif // include guard
