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
#include <cassert>
#include <cmath>
#include <iostream>
#include "QuadGenerator.hpp"
#include "TwoDLibException.hpp"

using namespace std;
using namespace TwoDLib;

QuadGenerator::QuadGenerator(const Quadrilateral& q, Uniform& uni):
_quadrilateral(q),
_uni(uni),
_tg1(q.Split().first,uni),
_tg2(q.Split().second,uni),
_f1(0),
_f2(0)
{
	double a1 = fabs(_tg1.Tri().SignedArea());
	double a2 = fabs(_tg2.Tri().SignedArea());
	_f1 = a1/(a1 + a2);
	_f2 = a2/(a1 + a2);
}

QuadGenerator::QuadGenerator(const QuadGenerator& q):
_quadrilateral(q._quadrilateral),
_uni(q._uni),
_tg1(_quadrilateral.Split().first,_uni),
_tg2(_quadrilateral.Split().second,_uni),
_f1(q._f1),
_f2(q._f2)
{
}


void QuadGenerator::Generate(vector<Point>* pvec) const {
	for (vector<TwoDLib::Point>::iterator it = pvec->begin(); it != pvec->end(); it++){
		GeneratePoint(it);
	}
}
