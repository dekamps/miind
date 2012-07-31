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

/**
 * @brief Interprolation rebinner is an important rebinning object
 *
 * Rebinning is necessary because the normal evolution of the population density takes
 * place in an expanding array, whose growth may not exceed a maxmimum. Once the maximum
 * is reached, rebinning must take place to fit the density profile in its original size.
 * This rebinner first smoothes the density profile around the reset bin, it then interpolates
 * the points in the new density profile using cubic spline interpolation on the old density profile.
 * Finally, the probability density that was taken from the reset bin is reintroduced at its new position.
 * The Rebin operation can take a pointer to AbstractZeroLeakEquations. If that pointer is non-zero, InterpolationRebinner
 * will assume that a refractive probability is maintained by AbstractZeroLeakEquations, and will rely on its
 * RefractiveProbability method to tell it how much. After rebinning, the total probability in the density profile
 * and the refractive probability will add up to one. As a consequence, when no pointer to AbstractZeroLeakEquations is given,
 * the integrated density (the sum over all bins) must be one.
 * InterpolationRebinner makes no assumptions whether, and does not check
 * if the profile before rebinning constitutes a density (i.e. integrates to one).
 */
class InterpolationRebinner: public AbstractRebinner {
public:
	/**
	 * default constructor
	 */
	InterpolationRebinner()=default;

	/**
	 * make non copyable
	 */
	InterpolationRebinner(const InterpolationRebinner&)=delete;
	/**
	 * make non copyable
	 */
	InterpolationRebinner& operator=(const InterpolationRebinner&)=delete;

	/**
	 * virtual Destructor
	 */
	virtual ~InterpolationRebinner();

	/**
	 * Configure: Here the a reference to the bin contenets, as well as parameters necessary for the rebinning are set
	 * @param array array with density profile that needs rebinning
	 * @param index_reversal_bin reversal bin
	 * @param index_reset_bin reset bin
	 * @param number_original_bins number of  bins before rebinning
	 * @param number_new_bins number of  bins after rebinning
	 */
	virtual void Configure(std::valarray<double>& array,
			Index index_reversal_bin, Index index_reset_bin,
			Number number_original_bins, Number number_new_bins);

	/**
	 * carry out rebinning as specified in configuration. The pointer to AbstractZeroLeakequations
	 * allows access to the refractive probability.
	 * passing in a null pointer is legal and leads to a no-operation
	 * @param p_zl A pointer to the AbstractZeroLeakEquations
	 */
	virtual void Rebin(zeroLeakEquations::AbstractZeroLeakEquations* p_zl=nullptr);

	/**
	 * Clone method
	 * @return A clone of InterpolationRebinner
	 */
	InterpolationRebinner* Clone() const;

	/**
	 * every rebinner has a name
	 * @return The name
	 */
	virtual std::string Name() const {
		return std::string("InterpolationRebinner");
	}

private:

	/**
	 * Purpose: the reset bin usually contains a spike. It is simply being replaced with
	 * the average density of its neighbours. The missing density will be reapplied after
	 * the rebinning
	 */
	void SmoothResetBin();

	/**
	 * Prepare the copy
	 */
	void PrepareLocalCopies();
	/**
	 * set the shrunk negative part of probability to zero
	 */
	void ResetOvershoot();
	/**
	 * Purpose: Locate new area of the new reset bin. If V_reset != V_reversal, they are different.
	 * Assumption: dv_before and dv_after have been calculated
	 * @return The new reset bin
	 */
	int IndexNewResetBin();
	/**
	 * Interpolate
	 */
	void Interpolate();

	/**
	 * Replaces the reset bin
	 * @param p_zl A pointer to the AbstractZeroLeakEquations
	 */
	void ReplaceResetBin(zeroLeakEquations::AbstractZeroLeakEquations* p_zl);
	/**
	 * rescale the array
	 */
	void RescaleAllProbability();

	/**
	 * Pointer to the iterator for interpolation
	 */
	gsl_interp_accel* _p_accelerator = gsl_interp_accel_alloc();
	/**
	 * Spline function
	 */
	gsl_spline* _p_spline = nullptr;

	/**
	 * vector of the x array
	 */
	std::vector<double> _x_array = std::vector<double>();
	/**
	 * vector of the y array
	 */
	std::vector<double> _y_array = std::vector<double>();
	/**
	 * The sum before rebinning
	 */
	double _sum_before = 0.0;
	/**
	 * Before rebinning
	 */
	double _dv_before = 0.0;
	/**
	 * After rebinning
	 */
	double _dv_after = 0.0;
	/**
	 * The pointer to the array
	 */
	std::valarray<double>* _p_array = nullptr;
	/**
	 * Index of the reversal bin
	 */
	int _index_reversal_bin = 0;
	/**
	 * index of the reset bin
	 */
	int _index_reset_bin = 0;
	/**
	 * number of  bins before rebinning
	 */
	Number _number_original_bins = 0;
	/**
	 * number of  bins after rebinning
	 */
	Number _number_new_bins = 0;
};} /* namespace rebinner */
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_REBINNER_INTERPOLATIONREBINNER_HPP_
