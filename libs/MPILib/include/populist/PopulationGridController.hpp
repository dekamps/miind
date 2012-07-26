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
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include "../NumtoolsLib/NumtoolsLib.h"
#include <MPILib/include/populist/AbstractRebinner.hpp>
#include <MPILib/include/populist/AbstractRateComputation.hpp>
#include <MPILib/include/populist/circulantSolvers/CirculantSolver.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>
#include <MPILib/include/populist/ZeroLeakBuilder.hpp>

using NumtoolsLib::D_Matrix;

namespace MPILib {
namespace populist {

//! PopulationGridController. Maintains the density profile.
//!
//! This class maintains the population density profile. Its main functions are
//! - to miantain the density profile
//! - to keep track of the instantaneous relation between bin number and membrane potential
//! - to maintain references to the circulant and the non-circulant solver
//! - to delegate the computation of the circulant and the non-circulant solution to respective solvers
//!
//! but not:
//!
//! - to compute the (non-)circulant solutions itelf, which is a task of the (Non)CirculantSolver classess
//! - to interpret input from other nodes, this is done by the PopulistAlgorithm, which makes input rates and
//!   efficacies available to the PopulistGridController

// Modification: 20-03-2009; Introduce template argument for decoding input parameters, this has become necessary
// by the introduction of the OneDMAlgorithm and some assumptions on how to convert input into parameters for the zero
// leak equation that were hardwired, have to be resolved by  the template argument.

template<class Weight>
class PopulationGridController {
public:

	//! constructor: receives references to the density array and the potential array which maintains the
	//! current membrane potential to which each density bin corresponds
	PopulationGridController( VALUE_REF

	const parameters::PopulationParameter&,

	const parameters::PopulistSpecificParameter&,

	//! density array
			valarray<double>&,

			//! potential array
			valarray<double>&,

			Number*,

			//!
			Rate*,

			//! log filestream
			ostringstream*);

	~PopulationGridController();

	//! Is newly configured whenever a new Configuration of a PopulationNode takes place, which in turn is triggered
	//! by a new configuration of the simulation
	void Configure(
	//! Starting time of simulation, etc
			const SimulationRunParameter&);

	//! This step calls the (Non)CirculantSolver and adapts the bin size aftwerwards. If it is time for rebinning,
	//! the AbstractRebinner will be called
	bool Evolve(Time, Time*, Rate*, const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector);

	//! Before the algorithm is carried out, the input parameters are calculated in all nodes so that
	//! the network as a whole is updated synchronously
	bool CollectExternalInput(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector);

	// EmbedGrid receives the initial density valarrays from the parent Algorithm
	// The state represented by these valarrays must be embed in the larger valarrays
	// of the Controller and also of the ParentAlgorithm.
	void EmbedGrid(Number, const valarray<double>&, const valarray<double>&,
			void* p_param = 0// Some algorithms need to pass parameters to their ZeroLeakEquations, PopulationGridController does want to know about their type

			);

	//! Number of current bins in the grid that are used to represent the density
	Number NumberOfCurrentBins() const;

	//! Current simulation time.
	Time CurrentTime() const;

	bool FillGammaZMatrix(Number, Number, Time);

	//! Convert an density bn index to a membrane potential
	Potential BinToCurrentPotential(Index) const;

	//! Convert a membrane potential to a density bin index
	Index CurrentPotentialToBin(Potential) const;

	//! converts a circulant index, which can have values 0, ..., n_circ into the corresponding state array values
	Index
	CirculantBinIndexToStateIndex(Index, const parameters::PopulationParameter&,
			const SpecialBins&) const;

private:

	PopulationGridController(const PopulationGridController&);

	PopulationGridController&
	operator=(const PopulationGridController&);

	bool IsReportPending(Time) const;

	void Rebin();
	void AddNewBins();
	void AdaptInterpretationArray();
	bool UpdateCheckSum();
	double NumberOfGrowingBins() const;
	bool IsTimeToRebin() const;
	Potential DeltaV() const;
	bool IsFinite() const;

	Time DeltaTimeNextBinAdded(Time) const;

	void RescaleInputParameters(Rate, Efficacy);

	Index IndexReversalBin() const;

	Index IndexOriginalResetBin() const;

	bool DefineRateArea();

	VALUE_MEMBER_REF

	Number _maximum_number_of_evolution_steps;
	Number _number_of_evolution_steps;

	Number _n_initial_bins;
	Number _n_bins_to_add;
	Number* _p_number_of_current_bins;
	Number _n_bins;

	Time _time_membrane_constant;
	Time _time_report;
	Time _time_next_report;
	Time _time_network_step;
	Time _time_current;

	SpecialBins _bins;
	parameters::PopulationParameter _par_pop;
	parameters::PopulistSpecificParameter _par_spec;

	double _f_current_scale;
	double _f_expansion_factor;

	Potential _delta_v;
	Potential _check_sum;

	Rate* _p_current_rate;
	// reference to the algorithm's grid state
	valarray<Density>& _array_state_reference;

	// local copy of _array_state, which need not be normalized during
	// evolution
	valarray<Density> _array_state;

	// interpretation array is only used during reports, hence
	// no local copy needed
	valarray<Potential>& _array_interpretation;
	boost::shared_ptr<AbstractRebinner> _p_rebinner;
	ostringstream* _p_stream;

	ZeroLeakBuilder _builder;
	boost::shared_ptr<AbstractZeroLeakEquations> _p_zl;

};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_POPULATIONGRIDCONTROLER_HPP_
