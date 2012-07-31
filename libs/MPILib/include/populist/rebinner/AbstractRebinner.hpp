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
#ifndef MPILIB_POPULIST_REBINNER_ABSTRACTREBINNER_HPP_
#define MPILIB_POPULIST_REBINNER_ABSTRACTREBINNER_HPP_

#include <valarray>
#include <MPILib/include/populist/zeroLeakEquations/AbstractZeroLeakEquations.hpp>
#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {
namespace populist {
namespace rebinner {

/**
 * @brief Abstract base class for rebinning algorithms.
 *
 * Rebinning algorithms serve to represent the density grid in the original grid, which is smaller
 * than the current grid, because grids are expanding over time. Various ways of rebinning are conceivable
 * and it may be necessary to compare different rebinning algorithms in the same program. The main simulation
 * step in PopulationGridController only needs to know that there is a rebinning algorithm.
 */
class AbstractRebinner {
public:

	/**
	 * default constructor
	 */
	AbstractRebinner()=default;
	/**
	 * virtual Destructor
	 */
	virtual ~AbstractRebinner() {};

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
			Number number_original_bins, Number number_new_bins)=0;

	/**
	 * every rebinner can do a rebin after it has been configured some rebinners need to
	 * take refractive probability into account
	 * @param p_zl
	 */
	virtual void Rebin(zeroLeakEquations::AbstractZeroLeakEquations* p_zl) = 0;

	/**
	 * Clone method
	 * @return A clone of AbstractRebinner
	 */
	virtual AbstractRebinner* Clone() const = 0;

	/**
	 * every rebinner has a name
	 * @return The name
	 */
	virtual std::string Name() const = 0;

protected:

	/**
	 * Scale the refractive
	 * @param scale The scale
	 * @param p_zl The equation
	 */
	void ScaleRefractive(double scale,
			zeroLeakEquations::AbstractZeroLeakEquations* p_zl);
};

} /* namespace rebinner */
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_REBINNER_ABSTRACTREBINNER_HPP_
