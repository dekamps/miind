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


#ifndef _CODE_LIBS_TRIANGLEGENERATOR_INCLUDE_GUARD
#define _CODE_LIBS_TRIANGLEGENERATOR_INCLUDE_GUARD
#include <iostream>

#include "Triangle.hpp"
#include "Uniform.hpp"

namespace TwoDLib {
	//! Class to generate uniformly random distributed points within the triangle, edges excluded

	//! Generates barycentric coordinates of the triangle. Relies on an external  Uniform distribution
	//! object.
	class TriangleGenerator {
	public:

		//! Constructor, accepts triangle. Seed sets the seed for the internal random generator.
		TriangleGenerator(const Triangle&, const Uniform&);

		void Generate(vector<Point>*) const;

		inline void GeneratePoint(vector<Point>::iterator) const;

		const Triangle& Tri() const {return _triangle;}

	private:

		const Uniform& _uni;
		const Triangle _triangle;

	};

	inline void TriangleGenerator::GeneratePoint(vector<Point>::iterator it) const {
		double u = 1.0;
		double v = 1.0;
		do {
			u = _uni.GenerateNext();
			v = _uni.GenerateNext();
		} 	while ( u + v >= 1.0);
		*it = _triangle._base + u*_triangle._span_1 + v*_triangle._span_2;
	}
}


#endif /* TRIANGLEGENERATOR_HPP_ */
