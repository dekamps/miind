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
#include "PolyGenerator.hpp"
#include "TwoDLibException.hpp"

using namespace std;
using namespace TwoDLib;

PolyGenerator::PolyGenerator(const Cell& cell, Uniform& uni):
_cell(cell),
_uni(uni)
{
	this->FillBoundingBox();
}

PolyGenerator::PolyGenerator(const PolyGenerator& q):
_cell(q._cell),
_uni(q._uni)
{
}

void PolyGenerator::GeneratePoint(vector<Point>::iterator it) const
{
	Point p;
	do {
		double f_x = _uni.GenerateNext();
		double f_y = _uni.GenerateNext();
		p[0] = _x_min + f_x*(_x_max - _x_min);
		p[1] = _y_min + f_y*(_y_max - _y_min);
	} while( ! _cell.IsInside(p));

	*it = p;
}

void PolyGenerator::Generate(vector<Point>* pvec) const {
	for (vector<TwoDLib::Point>::iterator it = pvec->begin(); it != pvec->end(); it++){
		GeneratePoint(it);
	}
}

void PolyGenerator::FillBoundingBox()
{
	double x_min = std::numeric_limits<double>::max();
	double y_min = std::numeric_limits<double>::max();
	double x_max = -std::numeric_limits<double>::max();
	double y_max = -std::numeric_limits<double>::max();

	const vector<Point>& vec_points = _cell.Points();
	for (const auto& point: vec_points)
	{
		if (point[0] < x_min) x_min = point[0];
		if (point[0] > x_max) x_max = point[0];
		if (point[1] < y_min) y_min = point[1];
		if (point[1] > y_max) y_max = point[1];
	}

	_x_min = x_min;
	_x_max = x_max;
	_y_min = y_min;
	_y_max = y_max;
}
