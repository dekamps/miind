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
#ifndef MPILIB_POPULIST_RATECOMPUTATION_ABSTRACTRATECOMPUTATION_HPP_
#define MPILIB_POPULIST_RATECOMPUTATION_ABSTRACTRATECOMPUTATION_HPP_

#include <valarray>

#include <MPILib/include/populist/parameters/InputParameterSet.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {
namespace populist {
namespace rateComputation {

/**
 * There are several methods to calculate a Population's firing rate from the population density
 * A virtual base class is provide, so that the methods can be exchanged in run time and the different
 * methods can be compared within a single simulation
 */
class AbstractRateComputation {
public:

	/**
	 * default constructor
	 */
	AbstractRateComputation()=default;

	/**
	 * Configuration of the RateComputation
	 * @param array_state state array (not const since a pointer to an element is needed, it should have been const otherwise)
	 * @param input_set current input to population
	 * @param par_population neuron parameters
	 * @param index_reversal index reversal bin
	 */
	virtual void Configure(std::valarray<Density>& array_state,
			const parameters::InputParameterSet& input_set,
			const parameters::PopulationParameter& par_population,
			Index index_reversal);

	/**
	 * virtual destructor
	 */
	virtual ~AbstractRateComputation() {};

	/**
	 * Clone method
	 * @return A clone of AbstractRateComputation
	 */
	virtual AbstractRateComputation* Clone() const = 0;
	/**
	 * Calculates the current rate
	 * @param nr_bins number current bins
	 * @return The current rate
	 */
	virtual Rate CalculateRate(Number) = 0;

protected:

	/**
	 * Defines the Rate Area
	 * @param v_lower The lower potential
	 */
	void DefineRateArea(Potential v_lower);

	/**
	 * Gets the Potential for the given bin
	 * @param index The index of the bin
	 * @return The current potential of the bin
	 */
	Potential BinToCurrentPotential(Index index);

	Index _index_reversal = 0;
	std::valarray<Density>* _p_array_state = nullptr;
	const parameters::InputParameterSet* _p_input_set = nullptr;
	parameters::PopulationParameter _par_population;
	std::valarray<Potential> _array_interpretation;

	Number _n_bins;
	Number _number_integration_area;
	Index _start_integration_area;

	Potential _delta_v_rel;
	Potential _delta_v_abs;

private:

};} /* namespace rateComputation*/
} /* namespace populist */
} /* namespace MPILib */
#endif // end of include guard MPILIB_POPULIST_RATECOMPUTATION_ABSTRACTRATECOMPUTATION_HPP_
