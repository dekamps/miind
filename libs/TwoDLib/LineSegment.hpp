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


#ifndef _CODE_LIBS_GEOMLIB_LINESEGMENT_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_LINESEGMENT_INCLUDE_GUARD

#include <cassert>
#include "Point.hpp"

namespace TwoDLib {
  /**
   * \brief A line segment is defined by a pair of points.
   *
   * The order of the points matters in calculations that involve the orientation of the line segment
   */
   

  class LineSegment {
  public:

    //!< Standard constructor
    LineSegment
    (
     const Point& p1, //!< First point
     const Point& p2  //!< Second point
     ):_p1(p1),_p2(p2){}


    //!< const operator access
    inline const Point& operator[](unsigned int i) const { assert( i < 2); return (i == 0) ? _p1 : _p2; }

    //!<  test whether two points are on the same side of a line
    bool ArePointsOnTheSameSide(const Point& p1, const Point& p2) const;

  private:

    const Point _p1;
    const Point _p2;
  };



}

#endif
