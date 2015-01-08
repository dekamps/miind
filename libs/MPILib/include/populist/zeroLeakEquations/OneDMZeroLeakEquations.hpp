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
#ifndef MPILIB_POPULIST_ZEROLEAKEQUATIONS_ONEDMZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_ZEROLEAKEQUATIONS_ONEDMZEROLEAKEQUATIONS_HPP_

#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>
#include <MPILib/include/DelayedConnection.hpp>
#include <MPILib/include/populist/zeroLeakEquations/AbstractZeroLeakEquations.hpp>
#include <MPILib/include/populist/zeroLeakEquations/ABConvertor.hpp>
#include <MPILib/include/populist/parameters/OneDMInputSetParameter.hpp>


namespace MPILib {
namespace populist {
namespace zeroLeakEquations {
class AbConvertor;

class OneDMZeroLeakEquations: public AbstractZeroLeakEquations {
public:

	/**
	 * default destructor
	 * @param n_bins reference to the variable keeping track of the current number of bins
	 * @param state reference to the state array
	 * @param check_sum reference to the check_sum
	 * @param bins reference to the special bins
	 * @param par_pop  serves now mainly to communicate t_s
	 * @param par_spec reference to PopulistSpecificParameter
	 * @param delta_v current potential interval covered by one bin
	 */
	OneDMZeroLeakEquations(Number& n_bins, std::valarray<Potential>& state,
			Potential& check_sum, SpecialBins& bins,
			parameters::PopulationParameter& par_pop,
			parameters::PopulistSpecificParameter& par_spec,
			Potential& delta_v);

	/**
	 * virtual destructor
	 */
	virtual ~OneDMZeroLeakEquations();
	/**
	 * Pass in whatever other parameters are needed. This is explicitly necessary for OneDMZeroLeakEquations
	 * @param p_void any pointer to a parameter
	 */
	virtual void Configure(void* p_void);
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
			const std::vector<NodeType>& typeVector) {
		_convertor.SortConnectionvector(nodeVector, weightVector, typeVector);
	}
	/**
	 * Every time step the input parameters must be adapted, even if the input doesn't
	 * change, because the are affected by LIF dynamics (see \ref population_algorithm).
	 */
	virtual void AdaptParameters() {
		_convertor.AdaptParameters();
	}
	/**
	 * Recalculates the solver parameters
	 */
	virtual void RecalculateSolverParameters() {
		_convertor.RecalculateSolverParameters();
	}
	/**
	 * Calculate the rate of the node
	 */
	virtual Time CalculateRate() const;

private:

	/**
	 * Initialise the gsl system
	 * @return the gsl system
	 */
	gsl_odeiv_system InitializeSystem() const;

	/**
	 * Reference to the number of bins
	 */
	Number& _n_bins;
	/**
	 * Pointer to the array state
	 */
	std::valarray<Potential>* _p_state;
	/**
	 * The gsl_odeiv_system
	 * moving frame prevents use of DVIntegrator
	 */
	gsl_odeiv_system _system;

	/**
	 * The ABConvertor
	 */
	ABConvertor _convertor;
	/**
	 * The max number
	 */
	Number _n_max;

	/**
	 * Pointer to gsl step
	 */
	gsl_odeiv_step* _p_step;
	/**
	 * Pointer to gsl control
	 */
	gsl_odeiv_control* _p_control;
	/**
	 * Pointer to gsl evolve
	 */
	gsl_odeiv_evolve* _p_evolve;
	/**
	 * The gsl_odeiv_system
	 */
	gsl_odeiv_system _sys;

	/**
	 * The OneDMInputSetParameter
	 */
	parameters::OneDMInputSetParameter _params;

};
} /* namespace zeroLeakEquations */
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ZEROLEAKEQUATIONS_ONEDMZEROLEAKEQUATIONS_HPP_
