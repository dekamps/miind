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

//! VArray
//! This calculates the one row of the matrix elements V_{kj}, which determine the circulant solution.
//! The full analytic expression from de Kamps (2006) is programmed, which is usefl benchmarking correctness
class VArray {

public:

	//! Defualt constructor
	VArray();

	//! Compute the array elements, based on the number of circulant bins, the number of non-circulant areas
	//! and for the time that is required
	bool FillArray(Number, // number of circulant bins
			Number, // number of non-circulant areas
			Time    // time over which the solution is required (tau)
			);

	//! V_{kj}, where indices the circulant bin and j the non-circulant area
	double V(Index, // k
			Index  // j
			) const;

private:

	bool FillArrayWithGarbage();

	bool CheckInNumbers(Number, Number);

	std::vector<double> _vector_array;
	Number _number_of_circulant_bins;
	Number _number_of_non_circulant_areas;
};

inline double VArray::V(Index index_circulant,
		Index index_non_circulant) const {
	assert( index_circulant < _number_of_circulant_bins);
	assert( index_non_circulant < _number_of_non_circulant_areas);
	return _vector_array[index_circulant + index_non_circulant];
}
} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_CIRCULANTSOLVERS_VARRAY_HPP_
