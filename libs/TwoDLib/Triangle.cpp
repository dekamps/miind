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
#include <iostream>
#include "Triangle.hpp"
#include "TwoDLibException.hpp"

using namespace std;
using namespace TwoDLib;

const unsigned int Triangle::_nr_points = 3;

Triangle::Triangle
(
	const Point& p1,
	const Point& p2,
	const Point& p3
):
Cell(VectorFromPoints(p1,p2,p3)),
_base(p1),
_span_1(0,0),
_span_2(0,0)
{
	_span_1 = p2 - p1;
	_span_2 = p3 - p1;
}

Triangle::Triangle
(
	const vector<double>& vec_v,
	const vector<double>& vec_w
):
Cell(vec_v,vec_w),
_base(_vec_points[0]),
_span_1(0,0),
_span_2(0,0)
{
	_span_1 = _vec_points[1]-_vec_points[0];
	_span_2 = _vec_points[2]-_vec_points[0];
}

Triangle::Triangle(const Triangle& r):
Cell(r),
_base(r._base),
_span_1(r._span_1),
_span_2(r._span_2)
{
}

Triangle::~Triangle()
{
}

vector<TwoDLib::Point> Triangle::VectorFromPoints(const Point& p1, const Point& p2, const Point& p3) const
{
	std::vector<TwoDLib::Point> vec(3);
	vec[0] = p1;
	vec[1] = p2;
	vec[2] = p3;
	return vec;
}
