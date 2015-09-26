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

#include "Response.hpp"
#include "GeomLibException.hpp"
#include <limits>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_errno.h>
#include <cassert>


namespace {
  const double SQRTPI  = sqrt(4*atan(1.0));
  const double EPSILON_RESPONSE_FUNCTION         = 1e-6; // absolute/relative error on spike response function
  const double MAXIMUM_INVERSE_RESPONSE_FUNCTION = 5;    // if upper limit is above this value, it might just as well be infinite
  const double F_ABS_REL_BOUNDARY = 1;                   // if upper limit > F_ABS_REL_BOUNDARY, use relative error

  double _Abru_negative( double x ){
	
		// Based on Abramowitz & Stegun
		// Rational approximation of erf(x)
		// equation 7.1.26, Dover edition
	
		assert (x < 0);
		x = -x;
	
		const double p =  0.3275911;
		const double a[5] = 
			{ 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429 };
	
		double t = 1/(1+p*x);
	
		double res = 0;
	
		res = (a[0] + ( a[1] + (a[2] + (a[3] + a[4]*t )*t )*t )*t )*t;
		return res;	
	}

	double _Abru_positive ( double x ){
		assert( x >= 0 );

		double erf = gsl_sf_erf(x);

		return exp(x*x)*(1 + erf);

	}

	double Abru( double x, void* ){
		if ( x >= 0 ) 
			return _Abru_positive(x);
		else
			return _Abru_negative(x);
	}



	double IntegralValue( double f_lower, double f_upper, double tau)
	{
		if ( f_upper < MAXIMUM_INVERSE_RESPONSE_FUNCTION )
		{
			double f_integral;
			double f_error;
			size_t number_of_evaluations;

			gsl_function F;

			F.function = &Abru;
			F.params   = 0;

			double f_absolute_error;
			double f_relative_error;

			// if f_upper > F_ABS_REL_BOUNDARY, the integral is big, and the
			// relative error is a more sensible measure
			if ( f_upper > F_ABS_REL_BOUNDARY)
			{
				f_absolute_error = 0;
				f_relative_error = EPSILON_RESPONSE_FUNCTION;
			}
			else
			{
				f_absolute_error = EPSILON_RESPONSE_FUNCTION;
				f_relative_error = 0;
			}

				int integration_result = 
					gsl_integration_qng
					(
						&F,
						f_lower,
						f_upper,
						f_absolute_error,
						f_relative_error,
						&f_integral,
						&f_error,
						&number_of_evaluations
					);

				if (integration_result != GSL_SUCCESS)
				  throw GeomLib::GeomLibException("Rate integrator problem");
				return SQRTPI*tau*f_integral;
			}
			else
			  return std::numeric_limits<double>::max();
		}

	double InterspikeTime
			(
				double theta,
				double mu,
				double sigma,
				double V_reset,
				double tau
			){

				double f_upper = (theta - mu)  /sigma;
				double f_lower = (V_reset - mu)/sigma;

				return IntegralValue(f_lower, f_upper, tau);		
			}

} // end of unnamed namespace


double GeomLib::ResponseFunction ( const GeomLib::ResponseParameter& par ){
	return 1/( par.tau_refractive + InterspikeTime(par.theta, par.mu, par.sigma, par.V_reset, par.tau) );
}

