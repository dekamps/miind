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
#ifndef MPILIB_POPULIST_CIRCULANTSOLVERS_VARRAY_HPP_
#define MPILIB_POPULIST_CIRCULANTSOLVERS_VARRAY_HPP_

#include <cassert>
#include <MPILib/include/TypeDefinitions.hpp>
#include <vector>
#include <algorithm>

namespace MPILib {
namespace populist {
namespace circulantSolvers {

/**
 * This calculates the one row of the matrix elements \f$ V_{kj}\f$, which determine the circulant solution.
 * The full analytic expression from de Kamps (2006) is programmed, which is useful benchmarking correctness
 */
class VArray {

public:

	/**
	 * Default constructor
	 */
	VArray();

	//!
	//!

	/**
	 * Compute the array elements, based on the number of circulant bins, the number of non-circulant areas
	 * and for the time that is required.
	 * @param number_circulant_bins number of circulant bins
	 * @param number_non_circulant_areas number of non-circulant areas
	 * @param tau  time over which the solution is required (tau)
	 */
	void FillArray(Number number_circulant_bins,
			Number number_non_circulant_areas, Time tau);

	//!
	/**
	 * \f$ V_{kj} \f$, where indices the circulant bin and j the non-circulant area
	 * @param index_circulant
	 * @param index_non_circulan
	 * @return
	 */
	double V(Index index_circulant, Index index_non_circulant) const;

private:

	/**
	 * Fill the VArray with garbage
	 */
	void FillArrayWithGarbage();

	/**
	 * Setter for number_of_circulant_bins and number_of_non_circulant_areas
	 * @param number_of_circulant_bins _number_of_circulant_bins is set to this
	 * @param number_of_non_circulant_areas _number_of_non_circulant_areas is set to this
	 */
	void CheckInNumbers(Number number_of_circulant_bins,
			Number number_of_non_circulant_areas);

	/**
	 * The vector array
	 */
	std::vector<double> _vector_array;
	/**
	 * The number of circulant bins
	 */
	Number _number_of_circulant_bins;
	/**
	 * the number of non circulant bins
	 */
	Number _number_of_non_circulant_areas;
};


} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_CIRCULANTSOLVERS_VARRAY_HPP_
