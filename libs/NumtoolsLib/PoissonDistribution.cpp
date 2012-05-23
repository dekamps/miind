// Copyright (c) 2005 - 2008 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "PoissonDistribution.h"

#include <cmath>
#include "BasicDefinitions.h"
#include <gsl/gsl_sf_gamma.h>

using namespace NumtoolsLib;


PoissonDistribution::PoissonDistribution(RandomGenerator& generator, double f_rate):
Distribution(generator),
_f_rate(f_rate),
_sq(0),
_alxm(0),
_g(0),
_oldm(-1.0)
{
}

double PoissonDistribution::NextSampleValue()
{

	double em,t,y;

	if ( _f_rate < 12.0) {
		if ( _f_rate != _oldm) {
			_oldm=_f_rate;
			_g=(float)(exp(-_f_rate));
		}
		em = -1;
		t=1.0;
		do {
			++em;
			t *= Distribution::NextSampleValue();
		} while (t > _g);
	} else {
		if (_f_rate != _oldm) {
			_oldm=_f_rate;
			_sq=(sqrt(2.0*_f_rate));
			_alxm=(log(_f_rate));
			_g=_f_rate*_alxm- gsl_sf_lngamma(_f_rate+1.0F);
		}
		do {
			do {
			
				y=(tan(PI*Distribution::NextSampleValue()));
				em=_sq*y+_f_rate;
			} while (em < 0.0);
		
			em=(floor(em));
			t=(0.9*(1.0+y*y)*exp(em*_alxm - gsl_sf_lngamma(em+1.0F)-_g));
		} while ( Distribution::NextSampleValue() > t);
	}
	return em;
}

