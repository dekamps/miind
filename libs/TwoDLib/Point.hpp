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


#ifndef _CODE_LIBS_GEOMLIB_POINT_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_POINT_INCLUDE_GUARD

#include <cassert>

namespace TwoDLib {
  /**
   * \brief Represents a 2D point as a (v,w) tuple.
   *
   */
   

  class Point {
  public:

	 //!<
	 Point():_v(0.0),_w(0.0){}

    //!< Standard constructor
    Point
    (
     double v, //!< First coordinate, often membrane potential
     double w //!< Second coordinate
     ):_v(v),_w(w){}

    Point(const Point& p):_v(p._v),_w(p._w){}

    //!< operator access
    inline double& operator[](unsigned int i){assert (i < 2); return (i == 0) ? _v : _w;}

    //! < const operator access
    inline const double& operator[](unsigned int i) const { assert( i < 2); return (i == 0) ? _v : _w; }

    Point& operator+=(const Point& p){ _v += p._v; _w += p._w; return *this; }

    Point& operator*=(double d){ _v *= d; _w*= d; return *this; }

	bool operator==(const Point& rhs) const { return (_v == rhs._v) && (_w == rhs._w); }

  private:

    double _v;
    double _w;

  };

  inline Point operator*(double d, const Point& p){ return Point(d*p[0],d*p[1]);}
  inline Point operator*(const Point& p, double d){ return Point(d*p[0],d*p[1]);}
  inline Point operator+(const Point& p1, const Point& p2){Point p(0,0);  p[0] = p1[0] + p2[0]; p[1] = p1[1] + p2[1]; return p;}
  inline Point operator-(const Point& p1, const Point& p2){Point p(0,0);  p[0] = p1[0] - p2[0]; p[1] = p1[1] - p2[1]; return p;}

  inline double dsquared(const Point& p1, const Point& p2){ return (p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] -p1[1]); }
}

#endif
