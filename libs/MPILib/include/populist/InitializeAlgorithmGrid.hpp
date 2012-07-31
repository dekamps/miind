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
#ifndef MPILIB_POPULIST_INITIALPOTENTIALVECTOR_HPP_
#define MPILIB_POPULIST_INITIALPOTENTIALVECTOR_HPP_

#include <vector>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/InitialDensityParameter.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>

namespace MPILib {
namespace populist {

class InitializeAlgorithmGrid {
public:
	/**
	 * Initialise the algoritm grid
	 * @param number_of_initial_bins The number of initial bins
	 * @param v_min The min
	 * @param parameter_population The PopulationParameter
	 * @param parameter_density The InitialDensityParameter
	 * @return The generated algorithm grid
	 */
	algorithm::AlgorithmGrid InitializeGrid(Number number_of_initial_bins,
			Potential v_min,
			const parameters::PopulationParameter& parameter_population,
			const parameters::InitialDensityParameter& parameter_density) const;

	/**
	 * Rebin the expansion factor
	 * @param number_initial_bins The number of initial bins
	 * @param v_min The min
	 * @param parameter_population The PopulationParameter
	 * @return the expansion factor
	 */
	double ExpansionFactorDoubleRebinner(Number number_initial_bins,
			Potential v_min,
			const parameters::PopulationParameter& parameter_population) const;
	/**
	 * Calculates the delta V
	 * @param number_of_initial_bins The number of initial bins
	 * @param v_min The min
	 * @param parameter_population The PopulationParameter
	 * @return the delta v
	 */
	Potential DeltaV(Number number_of_initial_bins, Potential v_min,
			const parameters::PopulationParameter& parameter_population) const;
	/**
	 * Calculates the reversal index
	 * @param number_of_initial_bins The number of initial bins
	 * @param v_min The min
	 * @param parameter_population The PopulationParameter
	 * @return the reversal index
	 */
	Index IndexReversal(Number number_of_initial_bins, Potential v_min,
			const parameters::PopulationParameter& parameter_population) const;

};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_INITIALPOTENTIALVECTOR_HPP_
