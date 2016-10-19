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


#ifndef _CODE_LIBS_GEOMLIB_COORDINATES_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_COORDINATES_INCLUDE_GUARD

#include <cassert>

namespace TwoDLib {
  /**
   * \brief Represents a Cell as a coordinate pair
   *
   */
   

  class Coordinates {
  public:

	 //!<
	 Coordinates():_i(0),_j(0){}

    //!< Standard constructor
    Coordinates
    (
     unsigned int i, //!< First coordinate, often membrane potential
     unsigned int j //!< Second coordinate
     ):_i(i),_j(j){}

    Coordinates(const Coordinates& p):_i(p._i),_j(p._j){}

    //!< operator access
    inline unsigned int & operator[](unsigned int i){assert (i < 2); return (i == 0) ? _i : _j;}

    //! < const operator access
    inline const unsigned int& operator[](unsigned int i) const { assert( i < 2); return (i == 0) ? _i : _j; }

    Coordinates& operator+=(const Coordinates& p){ _i += p._i; _j += p._j; return *this; }


  private:

    unsigned int _i;
    unsigned int _j;

  };

  inline Coordinates operator+(const Coordinates& p1, const Coordinates& p2){Coordinates p(0,0);  p[0] = p1[0] + p2[0]; p[1] = p1[1] + p2[1]; return p;}
  inline Coordinates operator-(const Coordinates& p1, const Coordinates& p2){Coordinates p(0,0);  p[0] = p1[0] - p2[0]; p[1] = p1[1] - p2[1]; return p;}
  inline bool operator==(const Coordinates& p1, const Coordinates& p2){ return p1[0] == p2[0] && p1[1] == p2[1]; }
  inline bool operator!=(const Coordinates& p1, const Coordinates& p2){return !(p1==p2);}
}

#endif
