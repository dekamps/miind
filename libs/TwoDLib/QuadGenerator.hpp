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


#ifndef _CODE_LIBS_QUADGENERATOR_INCLUDE_GUARD
#define _CODE_LIBS_QUADGENERATOR_INCLUDE_GUARD
#include <iostream>

#include "Quadrilateral.hpp"
#include "TriangleGenerator.hpp"
#include "Uniform.hpp"

namespace TwoDLib {
	//! Class to generate uniformly random distributed points within the quadrilateral, edges excluded

	class QuadGenerator {
	public:

		//! Constructor, accepts triangle. Seed sets the seed for the internal random generator.
		QuadGenerator(const Quadrilateral&, Uniform&);

		//! copy construction is OK, assignment is not
		QuadGenerator(const QuadGenerator&);

		void Generate(vector<Point>*) const;

		inline void GeneratePoint(vector<Point>::iterator) const;

	private:

		QuadGenerator& operator=(const QuadGenerator&);

		const Quadrilateral _quadrilateral;
		Uniform&			_uni;
		TriangleGenerator   _tg1;
		TriangleGenerator   _tg2;

		double	_f1;
		double	_f2;
	};

	inline void QuadGenerator::GeneratePoint(vector<Point>::iterator it) const {
		double f = _uni.GenerateNext();
		if (f < _f1)
			_tg1.GeneratePoint(it);
		else
			_tg2.GeneratePoint(it);
	}
}


#endif /* TRIANGLEGENERATOR_HPP_ */
