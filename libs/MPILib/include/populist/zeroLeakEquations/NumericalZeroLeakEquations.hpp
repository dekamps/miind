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
#ifndef MPILIB_POPULIST_ZEROLEAKEQUATIONS_NUMERICALZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_ZEROLEAKEQUATIONS_NUMERICALZEROLEAKEQUATIONS_HPP_

#include <memory>
#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/populist/zeroLeakEquations/AbstractZeroLeakEquations.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/zeroLeakEquations/LIFConvertor.hpp>
#include <MPILib/include/populist/parameters/NumericalZeroLeakParameter.hpp>
#include <MPILib/include/populist/ProbabilityQueue.hpp>
#include <MPILib/include/populist/rateComputation/AbstractRateComputation.hpp>

using NumtoolsLib::ExStateDVIntegrator;

namespace MPILib {
namespace populist {
namespace zeroLeakEquations {
/**
 * @brief Provides a numerical solution for the zero leak equations.
 */
class NumericalZeroLeakEquations: public AbstractZeroLeakEquations {
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
	NumericalZeroLeakEquations(Number& n_bins,
			std::valarray<Potential>& array_state, Potential& check_sum,
			SpecialBins& bins, parameters::PopulationParameter& par_pop,
			parameters::PopulistSpecificParameter& par_spec,
			Potential& delta_v);

	virtual ~NumericalZeroLeakEquations() {
	}

	/**
	 * Pass in whatever other parameters are needed. This is explicitly necessary for OneDMZeroLeakEquations
	 * @param p_void any pointer to a parameter
	 */
	virtual void Configure(void* p_void);

	/**
	 * Every Evolve step (but not every time step, see below), the input parameters must be updated
	 * @param nodeVector The vector which stores the Rates of the precursor nodes
	 * @param weightVector The vector which stores the Weights of the precursor nodes
	 * @param typeVector The vector which stores the NodeTypes of the precursor nodes
	 */
	virtual void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector);

	/**
	 * Every time step the input parameters must be adapted, even if the input doesn't
	 * change, because the are affected by LIF dynamics (see \ref population_algorithm).
	 */
	virtual void AdaptParameters();
	/**
	 * Recalculates the solver parameters
	 */
	virtual void RecalculateSolverParameters();
	/**
	 * Given input parameters, derived classes are free to implement their own solution for ZeroLeakEquations
	 * @param time The time
	 */
	virtual void Apply(Time time);
	/**
	 * Calculate the rate of the node
	 */
	Rate CalculateRate() const;

	/** Some  AbstractZeroLeakEquations have derived classes which keep track of refractive probability.
	 * These derived classes can overload this method, and make this amount available. For example,
	 * when rebinning this probability must be taken into account. See, e.g. RefractiveCirculantSolver.
	 * overload to account for refractive probability
	 */
	virtual Probability RefractiveProbability() const {
		return _queue.TotalProbability();
	}

private:
	/**
	 * Initialize the integrators
	 */
	void InitializeIntegrators();
	/**
	 * Push the stamped measure of probability on the queue
	 * @param t the time of the stamped measure of probability
	 * @param before The before parameter
	 */
	void PushOnQueue(Time t, double before);
	/**
	 * Pop the stamped measure of probability from the queue
	 * @param t the time of the stamped measure of probability
	 */
	void PopFromQueue(Time t);

	/**
	 * the current time
	 */
	Time _time_current = 0;
	/**
	 * A pointer to the number of bins
	 */
	Number* _p_n_bins;
	/**
	 * a pointer to the PopulationParameter
	 */
	parameters::PopulationParameter* _p_par_pop;
	/**
	 * a pointer to the Array state
	 */
	valarray<Potential>* _p_array_state;
	/**
	 * a pointer to check sum
	 */
	Potential* _p_check_sum;
	/**
	 * The LIFConvertor
	 */
	LIFConvertor _convertor;
	/**
	 * unique ptr to the AbstractRateComputation
	 */
	std::unique_ptr<rateComputation::AbstractRateComputation> _p_rate_calc;
	/**
	 * unique ptr to the ExStateDVIntegrator
	 */
	std::unique_ptr<ExStateDVIntegrator<parameters::NumericalZeroLeakParameter> > _p_integrator;
	/**
	 * unique ptr to the ExStateDVIntegrator
	 */
	std::unique_ptr<ExStateDVIntegrator<parameters::NumericalZeroLeakParameter> > _p_reset;
	/**
	 * The NumericalZeroLeakParameter parameters
	 */
	parameters::NumericalZeroLeakParameter _parameter;
	/**
	 * The ProbabilityQueue
	 */
	ProbabilityQueue _queue;
};
} /* namespace zeroLeakEquations */
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ZEROLEAKEQUATIONS_NUMERICALZEROLEAKEQUATIONS_HPP_
