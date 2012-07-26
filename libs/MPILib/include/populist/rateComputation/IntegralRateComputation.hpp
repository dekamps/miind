// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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
#ifndef MPILIB_POPULIST_RATECOMPUTATION_INTEGRALRATECOMPUTATION_HPP_
#define MPILIB_POPULIST_RATECOMPUTATION_INTEGRALRATECOMPUTATION_HPP_

#include <valarray>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <MPILib/include/populist/rateComputation/AbstractRateComputation.hpp>


namespace MPILib {
namespace populist {
namespace rateComputation{


//! IntegralRateComputation
//! Computes the firing rate of a population from the density profile, using an integral method:
//! \nu = \int^ \rho(v) dv
class IntegralRateComputation: public AbstractRateComputation {
public:

	//! constructor
	IntegralRateComputation();

	//! configuring gives access to density profile, the input parameters (effective efficacy and variance of eff. eff.)
	//! and the neuron parameters
	virtual void Configure(std::valarray<Density>&,	//! density valarray
			const parameters::InputParameterSet&, const parameters::PopulationParameter&, Index);

	virtual ~IntegralRateComputation();

	virtual IntegralRateComputation* Clone() const;

	virtual Rate CalculateRate(Number    // number current bins,
			);

private:

	gsl_interp_accel* _p_accelerator = nullptr;                //
	gsl_integration_workspace* _p_workspace = nullptr;           // need to be initialized

};
} /* namespace rateComputation*/
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_RATECOMPUTATION_INTEGRALRATECOMPUTATION_HPP_
