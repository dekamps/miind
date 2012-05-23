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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_LIBS_DYNAMICLIB_SPATIALPOSITION_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_SPATIALPOSITION_INCLUDE_GUARD

#include <limits>
#include <iostream>


using std::ostream;

namespace DynamicLib {


	struct SpatialPosition {

		SpatialPosition():
		_x(0),
		_y(0),
		_z(0),
		_f(0)
		{
		}

		SpatialPosition
		(
			float x,
			float y,
			float z,
			float f = 0
		):
		_x(x),
		_y(y),
		_z(z),
		_f(f)
		{
		}



		SpatialPosition& operator+=(const SpatialPosition& pos)
		{
			_x += pos._x;
			_y += pos._y;
			_z += pos._z;
			_f += pos._f;

			return *this;
		}

		float _x;
		float _y;
		float _z;
		float _f;

	};

	inline ostream& operator<<(ostream& s, const SpatialPosition& pos)
	{
		s << pos._x << " " << pos._y << " " << pos._z << " "  << pos._f << "\n";
		return s;
	}

	//! conventional comparison
	inline bool operator==(const SpatialPosition& p1, const SpatialPosition& p2)
	{
		return ( (p1._x == p2._x) && (p1._y == p2._y) && (p1._z == p2._z) && (p1._f == p2._f) );
	}

	//! inequality
	inline bool operator!=(const SpatialPosition& p1, const SpatialPosition& p2)
	{
		return ! (p1 == p2);
	}
}

#endif // include guard
