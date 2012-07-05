// Copyright (c) 2005 - 2011 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef MPILIB_POPULIST_POPOULATIONALGORITHM_CODE_HPP_
#define MPILIB_POPULIST_POPOULATIONALGORITHM_CODE_HPP_

#include <MPILib/include/populist/PopulationAlgorithm.hpp>
#include <MPILib/include/populist/ConnectionSquaredProduct.hpp>
#include <MPILib/include/populist/PopulationAlgorithm.hpp>
#include <MPILib/include/populist/InitializeAlgorithmGrid.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/populist/PopulistParameter.hpp>
#include <vector>

namespace MPILib {
namespace populist {

template<class Weight>
PopulationAlgorithm_<Weight>::PopulationAlgorithm_(
		const PopulistParameter& par_populist) :
		algorithm::AlgorithmInterface<PopulationConnection>(0), _parameter_population(
				par_populist._par_pop), _parameter_specific(
				par_populist._par_spec), _grid(
				AlgorithmGrid(_parameter_specific.MaxNumGridPoints())), _controller_grid(
#ifdef _INVESTIGATE_ALGORITHM
				_vec_value,
#endif
				_parameter_population, _parameter_specific,
				AbstractAlgorithm<Weight>::ArrayState(_grid),
				AbstractAlgorithm<Weight>::ArrayInterpretation(_grid),
				&AbstractAlgorithm<Weight>::StateSize(_grid), &_current_rate,
				&_stream_log), _current_time(0), _current_rate(0) {
	Embed();
}

template<class Weight>
PopulationAlgorithm_<Weight>::PopulationAlgorithm_(istream& s) :
		AbstractAlgorithm<PopulationConnection>(0), _parameter_population(
				ParPopFromStream(s)), _parameter_specific(ParSpecFromStream(s)), _grid(
				AlgorithmGrid(_parameter_specific.MaxNumGridPoints())), _controller_grid(
#ifdef _INVESTIGATE_ALGORITHM
				_vec_value,
#endif
				_parameter_population, _parameter_specific,
				AbstractAlgorithm<Weight>::ArrayState(_grid),
				AbstractAlgorithm<Weight>::ArrayInterpretation(_grid),
				&AbstractAlgorithm<Weight>::StateSize(_grid), &_current_rate,
				&_stream_log), _current_time(0), _current_rate(0) {
	_stream_log << "Running with ZeroLeakEquations: "
			<< _parameter_specific.ZeroLeakName() << "\n";
	_stream_log << "Running with Circulant: "
			<< _parameter_specific.CirculantName() << "\n";
	_stream_log << "Running with NonCirculant"
			<< _parameter_specific.NonCirculantName() << "\n";
	Embed();
	StripFooter(s);
}

template<class Weight>
PopulationAlgorithm_<Weight>::PopulationAlgorithm_(
		const PopulationAlgorithm_<Weight>& algorithm) :
		algorithm::AlgorithmInterface<PopulationConnection>(algorithm), _parameter_population(
				algorithm._parameter_population), _parameter_specific(
				algorithm._parameter_specific), _grid(
				AlgorithmGrid(_parameter_specific.MaxNumGridPoints())), _controller_grid(
#ifdef _INVESTIGATE_ALGORITHM
				_vec_value,
#endif
				_parameter_population, _parameter_specific,
				AbstractAlgorithm<Weight>::ArrayState(_grid),
				AbstractAlgorithm<Weight>::ArrayInterpretation(_grid),
				&AbstractAlgorithm<Weight>::StateSize(_grid), &_current_rate,
				&_stream_log), _current_time(0), _current_rate(0) {
	Embed();
}

template<class Weight>
void PopulationAlgorithm_<Weight>::Embed() {
	// at this point the local grid is big enough to accomodate expansion.

	// now the InitialDensity prescription together with V_min and the user-specified
	// number of bins are used to create an initial grid, which has a size given
	// by the initial number of bins
	InitializeAlgorithmGrid init;
	AlgorithmGrid grid_initial = init.InitializeGrid(
			_parameter_specific.NrGridInitial(), _parameter_specific.VMin(),
			_parameter_population, _parameter_specific.InitialDensity());
	// The initial grid now must be embedded in the local grid, which is generally larger.
	// In general the local grid of the PopulationAlgorithm is out of sync with the
	// valarrays maintained by the controller and it is only updated when a Report is due.
	// So this is done by the controller, since it maintans a local copy of the relevant
	// valarrays and needs insert the initial grid in the right sized valarray anyway.
	_controller_grid.EmbedGrid(_parameter_specific.NrGridInitial(),
			this->ArrayState(grid_initial), // only Algorithms can unpack the valarrays
			this->ArrayInterpretation(grid_initial) // so they must be sent to the controller
					);
}

template<class Weight>
void PopulationAlgorithm_<Weight>::evolveNodeState(
		const std::vector<Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector, Time time,
		const std::vector<NodeType>& typeVector) {

	bool b_return = _controller_grid.Evolve(time, &_current_time,
			&_current_rate, iter_begin, iter_end);

	return b_return;
}

template<class Weight>
void PopulationAlgorithm_<Weight>::prepareEvolve(
		const std::vector<Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector,
		const std::vector<NodeType>& typeVector) {
	_controller_grid.CollectExternalInput(iter_begin, iter_end);

}

template<class Weight>
PopulationAlgorithm_<Weight>::~PopulationAlgorithm_() {
}

template<class Weight>
void PopulationAlgorithm_<Weight>::configure(
		const SimulationRunParameter& par_run) {

	// The grid controller is configured with each new configuration of the network.
	_controller_grid.Configure(par_run);

	// write information in the log file with regard to algorithm settings
	WriteConfigurationToLog();

}

template<class Weight>
AlgorithmGrid PopulationAlgorithm_<Weight>::getGrid() const {
	return _grid;
}

template<class Weight>
Rate PopulationAlgorithm_<Weight>::getCurrentRate() const {
	return _current_rate;
}

template<class Weight>
Time PopulationAlgorithm_<Weight>::getCurrentTime() const {
	return _current_time;
}



template<class Weight>
Potential PopulationAlgorithm_<Weight>::BinToCurrentPotential(
		Index index) const {
	return _controller_grid.BinToCurrentPotential(index);
}

template<class Weight>
Index PopulationAlgorithm_<Weight>::CurrentPotentialToBin(Potential v) const {
	return _controller_grid.CurrentPotentialToBin(v);
}

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_POPOULATIONALGORITHM_CODE_HPP_
