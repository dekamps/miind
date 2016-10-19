// Copyright (c) 2005 - 2015 Marc de Kamps
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
#ifndef _CODE_LIBS_TWODLIB_FIDUCIALELEMENT_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_FIDUCIALELEMENT_INCLUDE_GUARD

#include "Mesh.hpp"

namespace TwoDLib {

	enum Overflow {CONTAIN, LEAK };

	typedef std::pair<Quadrilateral,Overflow> ProtoFiducial;

	struct FiducialElement {

		const Mesh*					_mesh;
		const Quadrilateral* 		_quad;
		Overflow					_overflow;
		const vector<Coordinates> 	_vec_coords;

		FiducialElement():_mesh(0),_quad(0),_vec_coords(0){}

		FiducialElement
		(
			const Mesh&               mesh,
			const Quadrilateral&      quad,
			Overflow				  overflow,
			const vector<Coordinates> vec_coords
		):
		_mesh(&mesh),
		_quad(&quad),
		_overflow(overflow),
		_vec_coords(vec_coords)
		{
		}
	};

	struct FidElementList {
	public:

		FidElementList():_vec_element(0){}

		FidElementList(const vector<FiducialElement>& vec_element):_vec_element(vec_element){}

		std::vector<TwoDLib::FiducialElement> _vec_element;
	};

}

# endif // include guard
