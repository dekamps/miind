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


#ifndef _CODE_LIBS_TWODLIB_CELL_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_CELL_INCLUDE_GUARD

#include <vector>
#include <utility>
#include "Point.hpp"

using std::pair;
using std::vector;

namespace TwoDLib {
  /**
   * \brief Base class for cells in Mesh.
   *
   * Cells in a Mesh are generally a Quadrilateral, but some regions 
   * of space are defined by elimination of other cells. 
   */
   

  class Cell {
  public:

	Cell
	(
		const vector<Point>&  //! vector of points defining the cell
	);

    /// The points must be offered such that they represent an oriented polygon
    Cell
    (
    	const vector<double>&, 	//!< vector containing v values of points
    	const vector<double>&	//!< vector containing w values if points
    );

    /// Copy constructor
    Cell(const Cell&);

    /// Virtual destructor
    virtual ~Cell() = 0;


    /// true of the point is inside the cell, false otherwise
    virtual bool IsInside(const Point&) const;

    double SignedArea() const { return _signed_area; }

    /// +1 for clockwise oriented, -1 for counter clockwise oriented
    int IsClockwise() const { return _b_is_clockwise; }

    /// return a reference to the Point instances making up this cell.
    const vector<Point>& Points() const {return _vec_points; }

    // return the centre of gravity of all points making up this cell
    Point Centroid() const { return _centroid; }

  protected:


    unsigned int _n_points;

    const vector<double> _vec_v;
    const vector<double> _vec_w;

    const vector<Point> _vec_points;

    double	_signed_area;
    int		_b_is_clockwise;
    Point	_centroid;

  private:

    std::pair< vector<double>, vector<double> > Vec(const vector<Point>&) const;
    TwoDLib::Point CalculateCentroid() const;

    double CalculateSignedArea() const;
    vector<Point> InitializePoints() const;


  };
}

#endif
