// Copyright (c) 2005 - 2014 Marc de Kamps
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
#include <MPILib/include/TypeDefinitions.hpp>
#include "Cell.hpp"
#include "modulo.hpp"

using namespace TwoDLib;

std::pair< vector<double>,vector<double> > Cell::Vec(const vector<Point>& vec_point) const {
	vector<double> vec_v(vec_point.size());
	vector<double> vec_w(vec_point.size());

	MPILib::Index i = 0;
	for (auto it = vec_point.begin(); it != vec_point.end(); it++, i++){
		vec_v[i] = vec_point[i][0];
		vec_w[i] = vec_point[i][1];
	}
	return std::pair<std::vector<double>, std::vector<double> >(vec_v,vec_w);
}
Cell::Cell
(
	const vector<TwoDLib::Point>& vec_points
):
_n_points(vec_points.size()),
_vec_v(Vec(vec_points).first),
_vec_w(Vec(vec_points).second),
_vec_points(vec_points),
_signed_area(CalculateSignedArea()),
_b_is_clockwise((_signed_area > 0) ? -1 : ((_signed_area < 0) ? 1 : 0)),
_centroid(CalculateCentroid())
{
}

Cell::Cell
(
	const vector<double>& vec_v,
	const vector<double>& vec_w
):
_n_points(vec_v.size()),
_vec_v(vec_v),
_vec_w(vec_w),
_vec_points(InitializePoints()),
_signed_area(CalculateSignedArea()),
_b_is_clockwise( (_signed_area > 0) ? -1 : ((_signed_area < 0) ? 1 : 0) ),
_centroid(CalculateCentroid())
{
	assert(vec_v.size() == vec_w.size());
}

Cell::Cell
(
		const Cell& c
):
_n_points(c._n_points),
_vec_v(c._vec_v),
_vec_w(c._vec_w),
_vec_points(c._vec_points),
_signed_area(c._signed_area),
_b_is_clockwise(c._b_is_clockwise),
_centroid(c._centroid)
{
}

Cell::~Cell()
{
}

TwoDLib::Point Cell::CalculateCentroid() const
{
	Point centroid(0.,0.);

	for(MPILib::Index i = 0; i < _n_points; i++){
		centroid[0] += _vec_v[i];
		centroid[1] += _vec_w[i];
	}
	centroid[0] /= static_cast<double>(_n_points);
	centroid[1] /= static_cast<double>(_n_points);

	return centroid;
}

vector<TwoDLib::Point> Cell::InitializePoints() const
{
	vector<Point> vec_ret;
	for (MPILib::Index i = 0; i < _n_points; i++){
		Point p (_vec_v[i],_vec_w[i]);
		vec_ret.push_back(p);
	}
	return vec_ret;
}

double Cell::CalculateSignedArea() const
{
	double area = 0.;

	for (MPILib::Index i = 0; i < _n_points -1; i++ ){ // last point needs special treatment
		area += _vec_points[i][0]*_vec_points[i+1][1] - _vec_points[i][1]*_vec_points[i+1][0];
	}

	// last point
	area += _vec_points[_n_points-1][0]*_vec_points[0][1] - _vec_points[_n_points-1][1]*_vec_points[0][0];

	return area/2.0;
}
/*
bool Cell::IsInside(const Point& p) const{

  for (MPILib::Index i = 0; i < _n_points; i++){
		double dx = _vec_points[modulo((i+1),_n_points)][0] - _vec_points[i][0];
		double dy = _vec_points[modulo((i+1),_n_points)][1] - _vec_points[i][1];

		double dpx = p[0] - _vec_points[i][0];
		double dpy = p[1] - _vec_points[i][1];
		double cp = (dpx*dy - dpy*dx)*_b_is_clockwise;
		if (cp < 0)
			return false;
	}

	return true;
}*/

bool Cell::IsInside(const Point& p) const
{
	// returns true if point p is inside this cell, false if outside. Uses ray tracing method,
	// and no longer relies on the assumption that the cell is convex.
	// Adapted from: http://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html

	  int counter = 0;
	  int N = _vec_points.size();
	  int i;
	  double xinters;
	  Point p1,p2;

	  p1 = _vec_points[0];
	  for (i=1;i<=N;i++) {
	    p2 = _vec_points[i % N];
	    if (p[1] > std::min(p1[1],p2[1])) {
	      if (p[1] <= std::max(p1[1],p2[1])) {
	        if (p[0] <= std::max(p1[0],p2[0])) {
	          if (p1[1] != p2[1]) {
	            xinters = (p[1]-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1])+p1[0];
	            if (p1[0] == p2[0] || p[0] <= xinters)
	              counter++;
	          }
	        }
	      }
	    }
	    p1 = p2;
	  }

	  if (counter % 2 == 0)
	    return false;
	  else
	    return true;
}
