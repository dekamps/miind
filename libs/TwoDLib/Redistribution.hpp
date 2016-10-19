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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_2DLIB_REDISTRIBUTION_INCLUDE_GUARD
#define _CODE_2DLIB_REDISTRIBUTION_INCLUDE_GUARD

#include <vector>
#include <iostream>
#include "Coordinates.hpp"

namespace TwoDLib {


	//! Auxiliary class to define redistribution of probability that has moved over threshold;
	struct Redistribution {
		Coordinates _from;
		Coordinates _to;
		double      _alpha;

		Redistribution():_from(Coordinates(0,0)),_to(Coordinates(0,0)),_alpha(1.0){}

		Redistribution(const Coordinates& from, const Coordinates& to, double alpha ):_from(from),_to(to),_alpha(alpha){}
	};

	//! Read a reset or a reversal mapping file and produce a mapping that can be used by Ode2DSystem
	std::vector<Redistribution> ReMapping(std::istream&);

	//! Write a reset or a reversal mapping to a stream. In the type argument you can specify reset or reversal
	void ToStream(const std::vector<Redistribution>&,std::ostream&, const std::string& type = "");
}


#endif //include guard
