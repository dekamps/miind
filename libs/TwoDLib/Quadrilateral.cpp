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
#include <sstream>
#include <NumtoolsLib/NumtoolsLib.h>
#include "LineSegment.hpp"
#include "Quadrilateral.hpp"
#include "TwoDLibException.hpp"

using namespace NumtoolsLib;
using namespace std;
using namespace TwoDLib;

const unsigned int Quadrilateral::_nr_points = 4;

bool Quadrilateral::SanityCheck() const
{
	if ( _vec_points[0][0] == _vec_points[1][0] && _vec_points[0][1] == _vec_points[1][1])
		return false;

	return true;
}



// Adapted from http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/1201356#1201356
// Returns 1 if the lines intersect, otherwise 0. In addition, if the lines
// intersect the intersection point may be stored in the floats i_x and i_y.
int get_line_intersection(float p0_x, float p0_y, float p1_x, float p1_y,
    float p2_x, float p2_y, float p3_x, float p3_y, float *i_x, float *i_y)
{

    float s1_x, s1_y, s2_x, s2_y;
    s1_x = p1_x - p0_x;     s1_y = p1_y - p0_y;
    s2_x = p3_x - p2_x;     s2_y = p3_y - p2_y;

    float s, t;
    float d = (-s2_x * s1_y + s1_x * s2_y);

    if (d == 0) // collinear or parallel, certainly not intersecting; this was not in the original code
    	return 0;

    // It is hard to tell what a reasonable minimum value for d is, mesh elements may be really small
    // and visual inspection of the mesh is important

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / d;
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / d;

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
    {
        // Collision detected
        if (i_x != NULL)
            *i_x = p0_x + (t * s1_x);
        if (i_y != NULL)
            *i_y = p0_y + (t * s1_y);
        return 1;
    }

    return 0; // No collision
}

bool Quadrilateral::IsSimple() const {

	// Run over all 4 edges. The only edge that needs to be tested is the non neighbouring one.
	// Unlike other functions, this one is particular to Quadrilaterals
	for (int i = 0; i < _n_points/2; i++){
		Point p_2 = _vec_points[modulo(i+2,_n_points)];
		Point p_3 = _vec_points[modulo(i+3,_n_points)];

		Point p_0 = _vec_points[i];
		Point p_1 = _vec_points[i+1];

		if (get_line_intersection(p_0[0],p_0[1],p_1[0],p_1[1],p_2[0],p_2[1],p_3[0],p_3[1],0,0) )
			return false;
	}

	return true;
}

bool Quadrilateral::IsConvex() const
{

	vector<double> cp(_n_points);
	for (int i = 0; i < _n_points; i++){
		double dx1 = _vec_points[modulo(i+1,_n_points)][0] - _vec_points[modulo(i,  _n_points)][0];
		double dx2 = _vec_points[modulo(i+2,_n_points)][0] - _vec_points[modulo(i+1,_n_points)][0];
		double dy1 = _vec_points[modulo(i+1,_n_points)][1] - _vec_points[modulo(i,  _n_points)][1];
		double dy2 = _vec_points[modulo(i+2,_n_points)][1] - _vec_points[modulo(i+1,_n_points)][1];
		cp[i] = dx1*dy2-dx2*dy1;
		if (i > 0)
			if ((cp[i]*cp[i-1] > 0) - (cp[i]*cp[i-1] < 0) < 0  )
				return false;
	}
	return true;
}

vector<TwoDLib::Point> Quadrilateral::VectorFromPoints(const Point& p1, const Point& p2, const Point& p3, const Point& p4) const
{
	std::vector<TwoDLib::Point> vec(Quadrilateral::_nr_points);
	vec[0] = p1;
	vec[1] = p2;
	vec[2] = p3;
	vec[3] = p4;

	return vec;
}

Quadrilateral::Quadrilateral
(
	const vector<TwoDLib::Point>& vec_point
):
Cell(vec_point)
{

	if (! this->SanityCheck())
			throw TwoDLibException(string("Sanity check failed in quadrilateral: "));


	if (! this->IsSimple()){
		std::ostringstream ost;
		for (const Point& p: _vec_points)
			ost << p[0] << "," << p[1] << ";";
		throw TwoDLibException(string("Quadrilateral is not simple.") + ost.str());
	}
}

Quadrilateral::Quadrilateral
(
	const Point& p1,
	const Point& p2,
	const Point& p3,
	const Point& p4
):
Cell(VectorFromPoints(p1,p2,p3,p4))
{
	if (! this->SanityCheck())
			throw TwoDLibException("Sanity check failed in quadrilateral.");

	if (! this->IsSimple()){
		std::ostringstream ost;
		for (const Point& p: _vec_points)
			ost << p[0] << ","<<  p[1] << ";";

		throw TwoDLibException(string("Quadrilateral is not simple.") + ost.str());
	}
}

Quadrilateral::Quadrilateral
(
		const vector<double>& vec_v,
		const vector<double>& vec_w
):
Cell(vec_v,vec_w)
{
	assert( vec_v.size() == Quadrilateral::_n_points);
	assert( vec_w.size() == Quadrilateral::_n_points);


	if (! this->SanityCheck())
			throw TwoDLibException("Sanity check failed in quadrilateral.");

	if (! this->IsSimple()){
		std::ostringstream ost;
		for (const Point& p: _vec_points)
			ost << p[0] << "," << p[1] << ";";

		throw TwoDLibException(string("Quadrilateral is not simple.") + ost.str());
	}
}

Quadrilateral::Quadrilateral(const Quadrilateral& quad):
Cell(quad)
{
	assert( _vec_v.size() == Quadrilateral::_n_points);
	assert( _vec_w.size() == Quadrilateral::_n_points);

	if (! this->SanityCheck())
			throw TwoDLibException("Sanity check failed in quadrilateral.");
	std::ostringstream ost;
	if (! this->IsSimple()){
		std::ostringstream ost;
		for (const Point& p: _vec_points)
			ost << p[0] << "," << p[1] << ";";

		throw TwoDLibException(string("Quadrilateral is not simple.") + ost.str());
	}
}



Quadrilateral::~Quadrilateral(){
}



pair<Triangle,Triangle> Quadrilateral::Split() const
{
	// if the quadrilateral is simple we can take any pair of opposing points, but
	// if not, one must check of the other two points are niot on the same side
	// of the connecting diagonal

	LineSegment l(_vec_points[0],_vec_points[2]);

	if (l.ArePointsOnTheSameSide(_vec_points[1],_vec_points[3]) == false){
		// we can use this diagonal

		Triangle t1(_vec_points[0], _vec_points[2],_vec_points[1]);
		Triangle t2(_vec_points[0], _vec_points[2],_vec_points[3]);

		return pair<Triangle,Triangle>(t1,t2);
	} else {
		Triangle t1(_vec_points[1],_vec_points[3],_vec_points[0]);
		Triangle t2(_vec_points[1],_vec_points[3],_vec_points[2]);

		return pair<Triangle,Triangle>(t1,t2);

	}

}

