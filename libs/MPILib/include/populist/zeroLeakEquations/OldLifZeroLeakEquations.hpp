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
#ifndef MPILIB_POPULIST_ZEROLEAKEQUATIONS_OLDLIFZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_ZEROLEAKEQUATIONS_OLDLIFZEROLEAKEQUATIONS_HPP_

#include <MPILib/include/populist/zeroLeakEquations/LIFZeroLeakEquations.hpp>
#include <MPILib/include/populist/rateComputation/AbstractRateComputation.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <memory>

namespace MPILib {
namespace populist {
namespace zeroLeakEquations {
/**
 * \deprecated DEPRECATED! In response to the discivery in (deKamps, 2006) that probability density
 * sometimes must be transported from one bin to a point between two bins a quick hack was devised,
 * essentially running the NonCirculantSolver twice, using the time to express the proportionality
 * of each bin. This is ugly and doubles simulation time.  OldZeroLeakEquations will not be available
 * for use in the XML version of MIIND.
 */
class OldLIFZeroLeakEquations: public LIFZeroLeakEquations {
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
	OldLIFZeroLeakEquations(Number& n_bins,
			std::valarray<Potential>& array_state, Potential& check_sum,
			SpecialBins& bins, parameters::PopulationParameter& par_pop,
			parameters::PopulistSpecificParameter& par_spec, Potential& delta_v,
			const circulantSolvers::AbstractCirculantSolver&,
			const nonCirculantSolvers::AbstractNonCirculantSolver&);

	virtual ~OldLIFZeroLeakEquations() {
	}

	/**
	 * No-op for OldLIFZeroLeakEquations
	 * @param p_void pointer to needed data
	 */
	virtual void Configure(void* p_void = 0);
	/**
	 * Given input parameters, derived classes are free to implement their own solution for ZeroLeakEquations
	 * @param time The time
	 */
	virtual void Apply(Time time);
	/**
	 * Every Evolve step (but not every time step, see below), the input parameters must be updated
	 * @param nodeVector The vector which stores the Rates of the precursor nodes
	 * @param weightVector The vector which stores the Weights of the precursor nodes
	 * @param typeVector The vector which stores the NodeTypes of the precursor nodes
	 */
	virtual void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<DelayedConnection>& weightVector,
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
	 * Calculate the rate of the node
	 */
	virtual Rate CalculateRate() const;

private:
	/**
	 * Apply the excitatory zero leak equation
	 * @param time The time point
	 */
	void ApplyZeroLeakEquationsAlphaExcitatory(Time time);
	/**
	 * Apply the inhibitory zero leak equation
	 * @param time The time point
	 */
	void ApplyZeroLeakEquationsAlphaInhibitory(Time time);
	/**
	 * the current time
	 */
	Time _time_current = 0.0;

};
} /* namespace zeroLeakEquations */
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ZEROLEAKEQUATIONS_OLDLIFZEROLEAKEQUATIONS_HPP_
