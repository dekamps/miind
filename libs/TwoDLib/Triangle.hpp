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


#ifndef _CODE_LIBS_TWODLIB_TRIANGLE_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_TRIANGLE_INCLUDE_GUARD

#include <vector>
#include "Cell.hpp"
#include "Point.hpp"

namespace TwoDLib {
/**
 * \brief A class supporting triangles
 *
 * A Triangle is defined by three points. The Quadrilateral is assumed to be
 * non-degenerate.
 */
	class Triangle : public Cell {
	public:

		Triangle
		(
			const Point&,
			const Point&,
			const Point&
		);


		Triangle
		(
				const vector<Point>&  //!< a vector of exactly three points
		);

		//! Constructor, requires two vectors of 3 coordinates each.
		Triangle
		(
				const vector<double>&, //!< a vector of exactly three v values
				const vector<double>&  //!< a vector of exactly three w values
		);

		//! Copy constructor
		Triangle(const Triangle&);

		//! Destructor
		virtual ~Triangle();

		static const unsigned int _nr_points;

		static int get_line_intersection(double p0_x, double p0_y, double p1_x, double p1_y,
		    double p2_x, double p2_y, double p3_x, double p3_y, double *i_x, double *i_y);
		static double get_overlap_area(const Triangle& t1, const Triangle& t2);
		static double sign (Point p1, Point p2, Point p3);
		static bool pointInTriangle(const Point& pt, const Triangle& t);
		static int orientation(Point p, Point q, Point r);
		static vector<Point> convexHull(const vector<Point>& points);

		void print() {
			std::cout << "Tri : " << _vec_points[0][0] << "," << _vec_points[0][1] << " | "
			<< _vec_points[1][0] << "," << _vec_points[1][1] << " | "
			<< _vec_points[2][0] << "," << _vec_points[2][1] << "\n";
		}

	private:

		friend class TriangleGenerator;

		vector<Point> VectorFromPoints(const Point& p1, const Point& p2, const Point& p3) const;

		Point _base;
		Point _span_1;
		Point _span_2;
	};
}

#endif /* TRIANGLE_HPP_ */
