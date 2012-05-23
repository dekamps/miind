// Copyright (c) 2005 - 2011 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_FITRATECOMPUTATION_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_FITRATECOMPUTATION_INCLUDE_GUARD

#include <gsl/gsl_multifit.h>
#include "AbstractRateComputation.h"

namespace PopulistLib {

	//! Fits a polynomial to the density profile near threshold and estimates the first and second derivative
	//! of the density profile, from which the output firing rate can be estamated.
	//!
	//! This method is general the preferred method. The IntegralRateComputation is only to be preferred is h is
	//! a relatively large. This is never the case in the diffusion approximation and therefore if input is emulated
	//! in the DoublePopulation mode, this method must be used. There are still anomalies for a population which has recently
	//! experienced large inhibitory input. The density profile is them moved away from threshold and negative values for the
	//! firing rate have been observed. This is caused by fitting a polynomial to an interval which is zero. This case
	//! must still be isolated. This class wraps GSL fitting data structures
	class FitRateComputation : public AbstractRateComputation {
	public:

		//! default constructor
		FitRateComputation();

		virtual void Configure
		(
			valarray<Density>&,			//!< state array
			const InputParameterSet&,	//!< current input to population
			const PopulationParameter&,	//!< neuron parameters
			Index						//!< index reversal bin
		);

		//! mandatory virtual destructor
		virtual ~FitRateComputation();

		//! mandatory clone operator
		virtual FitRateComputation* Clone() const;

		virtual Rate CalculateRate
		(
			Number    // number current bins,
		);

	private:

		bool IsSingleInput() const;

		void Alloc();
		void Free();

		void SetUpFittingMatrices();

		Rate ExtractRateFromFit();

		Number		_n_points;

		gsl_matrix*	_X; 
		gsl_matrix*	_cov;
		gsl_vector*	_c;

		// Data vector doesn't need to be allocated
		gsl_vector	_y;

		gsl_multifit_linear_workspace* _p_work;

	}; // end of FitRateComputation

}


#endif // include guard
