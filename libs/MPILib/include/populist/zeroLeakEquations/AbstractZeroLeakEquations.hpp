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
#ifndef MPILIB_POPULIST_ABSTRACTZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_ABSTRACTZEROLEAKEQUATIONS_HPP_

#include <MPILib/include/populist/zeroLeakEquations/SpecialBins.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>
#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/populist/circulantSolvers/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/parameters/InputParameterSet.hpp>
#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/NodeType.hpp>

namespace MPILib {
namespace populist {

/**
 * @brief A solver for the zero leak master equations in the PopulationAlgorithm.
 *
 * PopulationAlgorithm models the combined effect from leaky-integrate-and-fire (LIF) dynamics
 * and Poisson input spike trains on individual neurons. The effects of LIF dynamics are
 * accounted for by maintaining the density in a PopulationGridController. The PopulationGridController
 * implements exponential shrinkage (LIF decay) by relabeling the potential density every
 * time step and by adding points. The principle is explained in \ref population_algorithm.
 * Every time step the M equation governing the Poisson statistics of input spike trains
 * must be executed. This is handled by ZeroLeakEquations.
 */
class AbstractZeroLeakEquations {
public:

	/**
	 * Constructor, giving access to most relevant state variables held by PopulationGridController
	 * @param n_bins reference to the current number of bins
	 * @param array_state reference to state array
	 * @param check_sum reference to the check sum variable
	 * @param bins reference to bins variable: reversal bin, reset bin, etc
	 * @param par_pop reference to the PopulationParameter
	 * @param par_spec reference to the PopulistSpecificParameter
	 * @param delta_v reference to the current scale variable
	 */
	AbstractZeroLeakEquations(VALUE_REF_INIT
	Number& n_bins, std::valarray<Potential>& array_state, Potential& check_sum,
			SpecialBins& bins, parameters::PopulationParameter& par_pop,
			parameters::PopulistSpecificParameter& par_spec,
			Potential& delta_v
			) :
			_array_state(array_state), _par_pop(par_pop), _par_spec(par_spec), _bins(
					bins) {
	}

	virtual ~AbstractZeroLeakEquations() {
	}
	;

	/**
	 * Pass in whatever other parameters are needed. This is explicitly necessary for OneDMZeroLeakEquations
	 * @param any pointer to a parameter
	 */
	virtual void Configure(void*) = 0;

	/**
	 * Given input parameters, derived classes are free to implement their own solution for ZeroLeakEquations
	 * @param The time
	 */
	virtual void Apply(Time) = 0;

	/**
	 * Every Evolve step (but not every time step, see below), the input parameters must be updated
	 * @param nodeVector The vector which stores the Rates of the precursor nodes
	 * @param weightVector The vector which stores the Weights of the precursor nodes
	 * @param typeVector The vector which stores the NodeTypes of the precursor nodes
	 */
	virtual void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector) = 0;

	/**
	 * Every time step the input parameters must be adapted, even if the input doesn't
	 * change, because the are affected by LIF dynamics (see \ref population_algorithm).
	 */
	virtual void AdaptParameters() = 0;

	/**
	 * @todo write description
	 */
	virtual void RecalculateSolverParameters() = 0;
	/**
	 * @todo write description
	 */
	virtual Rate CalculateRate() const = 0;

	/** Some  AbstractZeroLeakEquations have derived classes which keep track of refractive probability.
	* These derived classes can overload this method, and make this amount available. For example,
	* when rebinning this probability must be taken into account. See, e.g. RefractiveCirculantSolver.
	*/
	virtual Probability RefractiveProbability() const {
		return 0.0;
	}

protected:
	/**
	 * @todo write description
	 */
	void SetInputParameter(const parameters::InputParameterSet& set) {
		_p_set = &set;
	}

	/**
	 * concrete instances of ZeroLeakEquations need to be able to manipulate mode
	 * @param mode
	 * @param solver
	 */
	void SetMode(CirculantMode mode, AbstractCirculantSolver& solver) {
		solver._mode = mode;
	}
	/**
	 * concrete instances of ZeroLeakEquations need to be able to manipulate mode
	 * @param mode
	 * @param solver
	 */	void SetMode(CirculantMode mode, AbstractNonCirculantSolver& solver) {
		solver._mode = mode;
	}

protected:

	std::valarray<double>& ArrayState() {
		return _array_state;
	}
	const parameters::PopulistSpecificParameter& ParSpec() const {
		return _par_spec;
	}
	const SpecialBins& Bins() const {
		return _bins;
	}
	const parameters::InputParameterSet& Set() const {
		return *_p_set;
	}

private:

	friend class AbstractRebinner;

	// Upon rebinning the refractive probability that an AbstractZeroLeakEquations subclass maintains must be
	// rescaled. This is only allowed to AbstractRebinners.
	virtual void ScaleRefractiveProbability(double) {
	}

	std::valarray<double>& _array_state;
	const parameters::PopulationParameter& _par_pop;
	const parameters::PopulistSpecificParameter& _par_spec;
	const SpecialBins& _bins;
	const parameters::InputParameterSet* _p_set = nullptr;
};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ABSTRACTZEROLEAKEQUATIONS_HPP_
