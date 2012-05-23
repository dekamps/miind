// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_VCHEBYSHEV_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_VCHEBYSHEV_INCLUDE_GUARD

//typedef unsigned int size_t;

#include <stddef.h>
#include <gsl/gsl_chebyshev.h>
#include "../NumtoolsLib/NumtoolsLib.h"

#include "BasicDefinitions.h"

using NumtoolsLib::QaDirty;


namespace PopulistLib {

inline double gsl_cheb_evaline (const gsl_cheb_series * cs, const double x)
{
  size_t i;
  double d1 = 0.0;
  double d2 = 0.0;

  double y = (2.0 * x - cs->a - cs->b) / (cs->b - cs->a);
  double y2 = 2.0 * y;

  for (i = cs->order; i >= 1; i--)
    {
      double temp = d1;
      d1 = y2 * d1 - d2 + cs->c[i];
      d2 = temp;
    }

  return y * d1 - d2 + 0.5 * cs->c[0];
}

	class VChebyshev {
	public:

		VChebyshev();
		~VChebyshev();

		Density V(Number, Number, Time);

	private:

		gsl_cheb_series* GenerateSeries(Index, Index);

		QaDirty<gsl_cheb_series*> _mat_p_series;

	};


	inline double VChebyshev::V(Index k, Index j, Time tau)
	{
		return  gsl_cheb_evaline(_mat_p_series(k,j),tau);
	}



}
#endif // include guard
