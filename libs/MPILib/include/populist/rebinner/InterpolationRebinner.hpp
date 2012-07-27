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

#ifndef MPILIB_POPULIST_REBINNER_INTERPOLATIONREBINNER_HPP_
#define MPILIB_POPULIST_REBINNER_INTERPOLATIONREBINNER_HPP_

#include <gsl/gsl_spline.h>
#include <MPILib/include/populist/rebinner/AbstractRebinner.hpp>
#include <MPILib/include/populist/zeroLeakEquations/AbstractZeroLeakEquations.hpp>

namespace MPILib {
namespace populist {
namespace rebinner {

//! Interprolation rebinner is an important rebinning object
//!
//! Rebinning is necessary because the normal evolution of the population density takes
//! place in an expanding array, whose growth may not exceed a maxmimum. Once the maximum
//! is reached, rebinning must take place to fit the density profile in its original size.
//! This rebinner first smooths the density profile around the reset bin, it then interpolates
//! the points in the new density profile using cubic spline interpolation on the old density profile.
//! Finally, the probability density that was taken from the reset bin is reintroduced at its new position.
//! The Rebin operation can take a pointer to AbstractZeroLeakEquations. If that pointer is non-zero, InterpolationRebinner
//! will assume that a refractive probability is maintained by AbstractZeroLeakEquations, and will rely on its
//! RefractiveProbability method to tell it how much. After rebinning, the total probability in the density profile
//! and the refractive probability will add up to one. As a consequence, when no pointer to AbstractZeroLeakEquations is given,
//! the integrated density (the sum over all bins) must be one.
//! InterpolationRebinner makes no assumptions wether, and does not check
//! if the profile before rebinning constitutes a density (i.e. intergrates to one).
class InterpolationRebinner: public AbstractRebinner {
public:

	InterpolationRebinner()=default;

	InterpolationRebinner(const InterpolationRebinner&)=delete;
	InterpolationRebinner& operator=(const InterpolationRebinner&)=delete;

	//! destructor
	virtual ~InterpolationRebinner();

	//! Configure
	virtual bool Configure(

			std::valarray<double>&,//!< array with density profile that needs rebinning
			Index,//!< reversal bin,
			Index,//!< reset bin
			Number,//!< number of  bins before rebinning
			Number//!< number of  bins after rebinning
	);

	//! carry out rebinning as specified in configuration. The pointer to AbstractZeroLeakequations allows access to the refractive probability.
	//! passing in a null pointer is legal and leads to a no-operation
	virtual bool Rebin(zeroLeakEquations::AbstractZeroLeakEquations* = 0);

	//! virtual constructor
	InterpolationRebinner* Clone() const;

	//! class name (for log file purposes)
	virtual std::string Name() const {
		return std::string("InterpolationRebinner");
	}

private:

	void SmoothResetBin();
	void PrepareLocalCopies();
	void ResetOvershoot();
	int IndexNewResetBin();
	void Interpolate();

	void ReplaceResetBin(zeroLeakEquations::AbstractZeroLeakEquations*);
	void RescaleAllProbability(zeroLeakEquations::AbstractZeroLeakEquations*);

	gsl_interp_accel* _p_accelerator = gsl_interp_accel_alloc();
	std::vector<double> _x_array = std::vector<double>();
	std::vector<double> _y_array = std::vector<double>();
	double _sum_before = 0.0;
	double _dv_before = 0.0;
	double _dv_after = 0.0;
	std::valarray<double>* _p_array = nullptr;
	int _index_reversal_bin = 0;
	int _index_reset_bin = 0;
	Number _number_original_bins = 0;
	Number _number_new_bins = 0;
	gsl_spline* _p_spline = nullptr;
};} /* namespace rebinner */
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_REBINNER_INTERPOLATIONREBINNER_HPP_
