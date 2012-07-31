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
#ifndef MPILIB_POPULIST_POPULATIONGRIDCONTROLER_HPP_
#define MPILIB_POPULIST_POPULATIONGRIDCONTROLER_HPP_

#include <valarray>
#include <memory>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/populist/rebinner/AbstractRebinner.hpp>
#include <MPILib/include/populist/rateComputation/AbstractRateComputation.hpp>
#include <MPILib/include/populist/circulantSolvers/CirculantSolver.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>
#include <MPILib/include/populist/ZeroLeakBuilder.hpp>

//using NumtoolsLib::D_Matrix;

namespace MPILib {
namespace populist {

/**
 * @brief  PopulationGridController. Maintains the density profile.
 *
 *
 *  This class maintains the population density profile. Its main functions are
 *  - to maintain the density profile
 *  - to keep track of the instantaneous relation between bin number and membrane potential
 *  - to maintain references to the circulant and the non-circulant solver
 *  - to delegate the computation of the circulant and the non-circulant solution to respective solvers
 *
 *  but not:
 *
 *  - to compute the (non-)circulant solutions itself, which is a task of the (Non)CirculantSolver classes
 *  - to interpret input from other nodes, this is done by the PopulistAlgorithm, which makes input rates and
 *    efficacies available to the PopulistGridController
 *
 * Modification: 20-03-2009; Introduce template argument for decoding input parameters, this has become necessary
 * by the introduction of the OneDMAlgorithm and some assumptions on how to convert input into parameters for the zero
 * leak equation that were hard wired, have to be resolved by  the template argument.
 */
template<class Weight>
class PopulationGridController {
public:

	/**
	 * constructor: receives references to the density array and the potential array which maintains the
	 * current membrane potential to which each density bin corresponds
	 * @param par_pop The PopulationParameter
	 * @param par_spec The PopulistSpecificParameter
	 * @param array_state The density array
	 * @param array_interpretation The potential array
	 * @param p_grid_size The size of the grid
	 * @param p_rate The rate
	 */
	PopulationGridController(const parameters::PopulationParameter& par_pop,
			const parameters::PopulistSpecificParameter& par_spec,
			valarray<double>& array_state,
			valarray<double>& array_interpretation, Number* p_grid_size,
			Rate* p_rate);

	/**
	 * The destructor
	 */
	~PopulationGridController() {
	}
	;
	/**
	 * delete copy constructor
	 */
	PopulationGridController(const PopulationGridController&)=delete;
	/**
	 * delete copy operator
	 */
	PopulationGridController& operator=(const PopulationGridController&)=delete;
	/**
	 * Is newly configured whenever a new Configuration of a PopulationNode takes place, which in turn is
	 * triggered by a new configuration of the simulation
	 * @param parameter_simulation Starting time of simulation, etc
	 */
	void Configure(const SimulationRunParameter& parameter_simulation);

	/**
	 * This step calls the (Non)CirculantSolver and adapts the bin size aftwerwards. If it is time for rebinning,
	 * the AbstractRebinner will be called
	 * @param time The time
	 * @param p_time_current The current time point
	 * @param p_rate_output The output rate
	 * @param nodeVector A vector of the Rates
	 * @param weightVector A vector of the weights
	 */
	void Evolve(Time time, Time* p_time_current, Rate* p_rate_output,
			const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector);

	/**
	 * Before the algorithm is carried out, the input parameters are calculated in all nodes so that
	 * the network as a whole is updated synchronously
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param typeVector Vector of the NodeTypes of the precursors
	 */
	bool CollectExternalInput(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector);

	/**
	 * EmbedGrid receives the initial density valarrays from the parent Algorithm
	 * The state represented by these valarrays must be embed in the larger valarrays
	 * of the Controller and also of the ParentAlgorithm.
	 * @param nr_initial_grid The initial grid size
	 * @param array_state_initial The initial state array
	 * @param array_interpretation_initial The initial interpretation array
	 * @param p_param Some algorithms need to pass parameters to their ZeroLeakEquations, PopulationGridController does want to know about their type
	 */
	void EmbedGrid(Number nr_initial_grid,
			const valarray<double>& array_state_initial,
			const valarray<double>& array_interpretation_initial,
			void* p_param = nullptr);

	/**
	 * Number of current bins in the grid that are used to represent the density
	 * @return The Number of current bins
	 */
	Number NumberOfCurrentBins() const;

	/**
	 * Getter for the current time
	 * @return The current time
	 */
	Time CurrentTime() const;

	/**
	 * Convert an density bin index to a membrane potential
	 * @param index The index of the density bin
	 * @return The membran potential
	 */
	Potential BinToCurrentPotential(Index index) const;

	/**
	 * Convert a membrane potential to a density bin index
	 * @param v The membran potential
	 * @return The index of the bin
	 */
	Index CurrentPotentialToBin(Potential v) const;

private:

	/**
	 * Is it time for the new report
	 * @param time the current time
	 * @return true if a report should be written
	 */
	bool IsReportPending(Time time) const;

	/**
	 * Rebin the array grid
	 */
	void Rebin();
	/**
	 * Add new bins to the array grid
	 */
	void AddNewBins();
	/**
	 * Adapt the Interpretation array
	 */
	void AdaptInterpretationArray();
	/**
	 * Update the check sum
	 */
	void UpdateCheckSum();

	/**
	 * Gets the number of growing bins
	 * @return The number of growing bins
	 */
	double NumberOfGrowingBins() const;
	/**
	 * rebin if the next step would take you past the boundary array
	 * @return true if one chould rebin
	 */
	bool IsTimeToRebin() const;
	/**
	 * Calculates the Potential
	 * @return The Potential
	 */
	Potential DeltaV() const;
	/**
	 * Check if it is finite
	 * @return true if it is finite
	 */
	bool IsFinite() const;

	/**
	 * Calculates the time difference to the next bin add
	 * @param time the current time
	 * @return The delta time
	 */
	Time DeltaTimeNextBinAdded(Time time) const;

	/**
	 * Getter for the reversal bin
	 * @return The reversal bin
	 */
	Index IndexReversalBin() const;
	/**
	 * Getter for the original reset bin
	 * @return The original reset bin
	 */
	Index IndexOriginalResetBin() const;

	/**
	 * The maximum number of evolution steps
	 */
	Number _maximum_number_of_evolution_steps;
	/**
	 * The current number of evolution steps
	 */
	Number _number_of_evolution_steps;

	/**
	 * The number of initial bins
	 */
	Number _n_initial_bins;
	/**
	 * The number of bins to add
	 */
	Number _n_bins_to_add;
	/**
	 * pointer to the current number of bins
	 */
	Number* _p_number_of_current_bins;
	/**
	 * The number of bins
	 */
	Number _n_bins;

	/**
	 * The time membrane constant
	 */
	Time _time_membrane_constant;
	/**
	 * The report time
	 */
	Time _time_report = 0.0;
	/**
	 * The time of the next report
	 */
	Time _time_next_report = 0.0;
	/**
	 * The time of the network step
	 */
	Time _time_network_step = 0.0;
	/**
	 * The current time point
	 */
	Time _time_current;

	/**
	 * The special bins
	 */
	zeroLeakEquations::SpecialBins _bins;
	/**
	 * The PopulationParameter
	 */
	parameters::PopulationParameter _par_pop;
	/**
	 * The PopulistSpecificParameter
	 */
	parameters::PopulistSpecificParameter _par_spec;

	/**
	 * The scale factor
	 */
	double _f_current_scale = 1.0;
	/**
	 * The expansion factor
	 */
	double _f_expansion_factor;

	/**
	 * The DeltaV
	 */
	Potential _delta_v = 0.0;
	/**
	 * The check sum
	 */
	Potential _check_sum;

	/**
	 * A pointer to the current rate
	 */
	Rate* _p_current_rate;
	/**
	 * reference to the algorithm's grid state
	 */
	std::valarray<Density>& _array_state_reference;

	/**
	 * local copy of _array_state, which need not be normalized during evolution
	 */
	std::valarray<Density> _array_state;

	/**
	 * interpretation array is only used during reports, hence no local copy needed
	 */
	std::valarray<Potential>& _array_interpretation;
	/**
	 * a shared pointer to the abstract Rebinner
	 */
	std::shared_ptr<rebinner::AbstractRebinner> _p_rebinner;

	/**
	 * The ZeroLeakBuilder
	 */
	ZeroLeakBuilder _builder;
	/**
	 * a shared pointer to the AbstractZeroLeakEquations
	 */
	std::shared_ptr<zeroLeakEquations::AbstractZeroLeakEquations> _p_zl;

};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_POPULATIONGRIDCONTROLER_HPP_
