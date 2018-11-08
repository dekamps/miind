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
#include <unordered_set>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace TwoDLib;

const unsigned int Triangle::_nr_points = 3;

int Triangle::get_line_intersection(double p0_x, double p0_y, double p1_x, double p1_y,
    double p2_x, double p2_y, double p3_x, double p3_y, double *i_x, double *i_y)
{
    double s1_x, s1_y, s2_x, s2_y;
    s1_x = p1_x - p0_x;     s1_y = p1_y - p0_y;
    s2_x = p3_x - p2_x;     s2_y = p3_y - p2_y;

    double s, t;
    double d = (-s2_x * s1_y + s1_x * s2_y);

    if (d == 0) { // collinear or parallel, certainly not intersecting; this was not in the original code
    	return 0;
    }

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

double Triangle::get_overlap_area(const Triangle& t1, const Triangle& t2) {
  //printf("T1 = %f,%f | %f,%f | %f,%f\n", t1._vec_points[0][0], t1._vec_points[0][1], t1._vec_points[1][0], t1._vec_points[1][1], t1._vec_points[2][0], t1._vec_points[2][1]);
  //printf("T2 = %f,%f | %f,%f | %f,%f\n", t2._vec_points[0][0], t2._vec_points[0][1], t2._vec_points[1][0], t2._vec_points[1][1], t2._vec_points[2][0], t2._vec_points[2][1]);
	unordered_set<Point> overlap_poly = unordered_set<Point>();

	// if any points in t2 are within t1 :
	for(int i=0; i<3; i++) {
		if(Triangle::pointInTriangle(t2._vec_points[i], t1))
			overlap_poly.insert(t2._vec_points[i]);
	}
	// If all points of t2 are in t1, just return t2's area.
	if(overlap_poly.size() == 3)
		return std::abs(t2.SignedArea());

	// if points in t1 are in t2, include those
	int t1_in_t2 = 0;
	for(int i=0; i<3; i++) {
		if(Triangle::pointInTriangle(t1._vec_points[i], t2)){
			t1_in_t2++;
			overlap_poly.insert(t1._vec_points[i]);
		}
	}

	// If all point of t1 are in t2, just return t1's area.
	if(t1_in_t2 == 3)
		return std::abs(t1.SignedArea());

	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) {
			double x,y;
			int i_mod = (i+1) % 3;
			int j_mod = (j+1) % 3;
			if(get_line_intersection(t1._vec_points[i][0], t1._vec_points[i][1], t1._vec_points[i_mod][0], t1._vec_points[i_mod][1],
																t2._vec_points[j][0], t2._vec_points[j][1], t2._vec_points[j_mod][0], t2._vec_points[j_mod][1],
																&x,&y)) {
        // printf("%f,%f - %f,%f | %f,%f - %f,%f : %f %f\n", t1._vec_points[i][0], t1._vec_points[i][1], t1._vec_points[i_mod][0], t1._vec_points[i_mod][1],
  			// 													t2._vec_points[j][0], t2._vec_points[j][1], t2._vec_points[j_mod][0], t2._vec_points[j_mod][1], x,y);
				overlap_poly.insert(Point(x,y));
			}
		}
	}

	vector<Triangle> triangles = vector<Triangle>();
  double area = 0;

  if(overlap_poly.size() > 2) {
    vector<Point> vec_overlap_poly = vector<Point>(overlap_poly.begin(), overlap_poly.end());

    vector<Point> vec_ordered_overlap = vector<Point>(vec_overlap_poly);
    if(vec_overlap_poly.size() > 3)
      vector<Point> vec_ordered_overlap = convexHull(vec_overlap_poly);

    if(vec_ordered_overlap.size() > 2) {
    	for(int i=0; i<static_cast<unsigned int>(vec_ordered_overlap.size() - 2); i++){
        triangles.push_back(Triangle(vec_ordered_overlap[0], vec_ordered_overlap[i+1], vec_ordered_overlap[i+2]));
    	}

    	for(Triangle t : triangles) {
        area += std::abs(t.SignedArea());
    	}
    }
  }

	return area;
}

int Triangle::orientation(Point p, Point q, Point r)
{
    int val = (q[1] - p[1]) * (r[0] - q[0]) -
              (q[0] - p[0]) * (r[1] - q[1]);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

// Prints convex hull of a set of n points.
vector<Point> Triangle::convexHull(const vector<Point>& points)
{
    // Initialize Result
    vector<Point> hull;

    // Find the leftmost point
    int l = 0;
    for (int i = 1; i < static_cast<unsigned int>(points.size()); i++)
        if (points[i][0] < points[l][0])
            l = i;

    vector<int> picked = vector<int>();
    // Start from leftmost point, keep moving counterclockwise
    // until reach the start point again.  This loop runs O(h)
    // times where h is number of points in result or output.
    int p = l, q;
    do
    {
        // Add current point to result
        hull.push_back(points[p]);
        picked.push_back(p);

        // Search for a point 'q' such that orientation(p, x,
        // q) is counterclockwise for all points 'x'. The idea
        // is to keep track of last visited most counterclock-
        // wise point in q. If any point 'i' is more counterclock-
        // wise than q, then update q.
        q = (p+1)%(static_cast<unsigned int>(points.size()));
        for (int i = 0; i < static_cast<unsigned int>(points.size()); i++)
        {
           // If i is more counterclockwise than current q, then
           // update q
           if (orientation(points[p], points[i], points[q]) == 2 && find(picked.begin(), picked.end(), i) == picked.end())
               q = i;
        }

        // Now q is the most counterclockwise with respect to p
        // Set p as q for next iteration, so that q is added to
        // result 'hull'
        p = q;

    } while (p != l);  // While we don't come to first point

    return hull;
}

double Triangle::sign (Point p1, Point p2, Point p3)
{
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
}

bool Triangle::pointInTriangle (const Point& pt, const Triangle& t)
{
    bool b1, b2, b3;

    b1 = sign(pt, t._vec_points[0], t._vec_points[1]) < 0;
    b2 = sign(pt, t._vec_points[1], t._vec_points[2]) < 0;
    b3 = sign(pt, t._vec_points[2], t._vec_points[0]) < 0;

    return ((b1 == b2) && (b2 == b3));
}

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
