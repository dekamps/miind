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

#ifndef MPILIB_POPULIST_ZEROLEAKBUILDER_HPP_
#define MPILIB_POPULIST_ZEROLEAKBUILDER_HPP_

#include <string>
#include <memory>
#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/populist/zeroLeakEquations/AbstractZeroLeakEquations.hpp>
#include <MPILib/include/populist/rateComputation/AbstractRateComputation.hpp>

namespace MPILib {
namespace populist {

class ZeroLeakBuilder {
public:

	/**
	 * default constructor
	 * @param n_bins reference to the current number of bins
	 * @param array_state reference to state array
	 * @param checksum reference to the check sum variable
	 * @param bins reference to the Special Bins
	 * @param par_pop reference to the PopulationParameter
	 * @param par_spec reference to the PopulistSpecificParameter
	 * @param delta_v reference to the current scale variable
	 */
	ZeroLeakBuilder(Number& n_bins, std::valarray<Potential>& array_state,
			Potential& checksum, zeroLeakEquations::SpecialBins& bins,
			parameters::PopulationParameter& par_pop,
			parameters::PopulistSpecificParameter& par_spec,
			Potential& delta_v);

	/**
	 * Constructs the zero Leak Equations
	 * @param zeroleakequations_name The name of the zeroleak equation
	 * @param circulant_solver_name The name of the circulant solver
	 * @param noncirculant_solver_name The name of the non circulant solver
	 * @return
	 */
	std::shared_ptr<zeroLeakEquations::AbstractZeroLeakEquations> GenerateZeroLeakEquations(
			const std::string& zeroleakequations_name,
			const std::string& circulant_solver_name,
			const std::string& noncirculant_solver_name);

private:

	/**
	 * reference to the current number of bins
	 */
	Number& _n_bins;
	/**
	 * reference to state array
	 */
	std::valarray<Potential>& _array_state;
	/**
	 * reference to the check sum variable
	 */
	Potential& _checksum;
	/**
	 * reference to the Special Bins
	 */
	zeroLeakEquations::SpecialBins& _bins;
	/**
	 * reference to the PopulationParameter
	 */
	parameters::PopulationParameter& _par_pop;
	/**
	 *  reference to the PopulistSpecificParameter
	 */
	parameters::PopulistSpecificParameter& _par_spec;
	/**
	 * reference to the current scale variable
	 */
	Potential& _delta_v;

};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ZEROLEAKBUILDER_HPP_
