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


#ifndef _CODE_LIBS_TWODLIB_QUADRILATERAL_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_QUADRILATERAL_INCLUDE_GUARD

#include <vector>
#include "Triangle.hpp"


using std::vector;

namespace TwoDLib {
  /**
   * \brief Base class for cells in Mesh.
   *
   * Cells in a Mesh are generally a Quadrilateral,
   * a Quadrilateral is defined by four points. The Quadrilateral is assumed to be
   * non-degenerate and simple (not self-intersecting). An attempt to create a complex Quadrilateral
   * will result in an exception.
   */


  class Quadrilateral : public Cell {
  public:

	//! Standard constructor accepting a vector of exactly four Point s
	Quadrilateral
	(
		const vector<Point>&
	);

    //! Standard constructor, takes exactly four Point instances.
	//! Mind the ordering layed out in
    Quadrilateral
    (
    	const vector<double>&, //!< vector or exactly four v values
    	const vector<double>&  //!< vector of exactly four w values
    );

    Quadrilateral
    (
    		const Point&,
       		const Point&,
       		const Point&,
       		const Point&
    );

    //! Copy constructor
    Quadrilateral(const Quadrilateral&);

    Quadrilateral(const Cell&);

    //! Virtual destructor
    virtual ~Quadrilateral();

    //! sign is positive for counter clockwise oriented quadrilaterals
    double SignedArea() const { return _signed_area; }

    //! Split into two triangles. For complex quadrilaterals this means the split must be done along the right diagonal
    std::pair<Triangle,Triangle> Split() const;

    static double get_overlap_area(const Quadrilateral& q1, const Quadrilateral& q2);

    static const unsigned int _nr_points;

  private:

	vector<Point> VectorFromPoints(const Point& p1, const Point& p2, const Point& p3, const Point& p4) const;


    bool IsConvex() const;
    bool IsSimple() const;

    bool SanityCheck() const;

  };
}

#endif
