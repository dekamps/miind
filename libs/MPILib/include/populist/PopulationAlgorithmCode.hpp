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
#ifndef MPILIB_POPULIST_POPOULATIONALGORITHM_CODE_HPP_
#define MPILIB_POPULIST_POPOULATIONALGORITHM_CODE_HPP_

#include <MPILib/include/populist/PopulationAlgorithm.hpp>
#include <MPILib/include/populist/PopulationAlgorithm.hpp>
#include <MPILib/include/populist/InitializeAlgorithmGrid.hpp>
#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/PopulistParameter.hpp>
#include <cassert>
#include <vector>

namespace MPILib {
namespace populist {

template<class Weight>
PopulationAlgorithm_<Weight>::PopulationAlgorithm_(
		const parameters::PopulistParameter& par_populist) :
		algorithm::AlgorithmInterface<OrnsteinUhlenbeckConnection>(), _parameter_population(
				par_populist._par_pop), _parameter_specific(
				par_populist._par_spec), _grid(
				algorithm::AlgorithmGrid(
						_parameter_specific.getMaxNumGridPoints())), _controller_grid(
#ifdef _INVESTIGATE_ALGORITHM
				_vec_value,
#endif
				_parameter_population, _parameter_specific,
				algorithm::AlgorithmInterface<Weight>::getArrayState(_grid),
				algorithm::AlgorithmInterface<Weight>::getArrayInterpretation(
						_grid),
				&algorithm::AlgorithmInterface<Weight>::getStateSize(_grid),
				&_current_rate) {
	Embed();
}


template<class Weight>
PopulationAlgorithm_<Weight>::PopulationAlgorithm_(
		const PopulationAlgorithm_<Weight>& algorithm) :
		algorithm::AlgorithmInterface<OrnsteinUhlenbeckConnection>(algorithm), _parameter_population(
				algorithm._parameter_population), _parameter_specific(
				algorithm._parameter_specific), _grid(
				algorithm::AlgorithmGrid(
						_parameter_specific.getMaxNumGridPoints())), _controller_grid(
#ifdef _INVESTIGATE_ALGORITHM
				_vec_value,
#endif
				_parameter_population, _parameter_specific,
				algorithm::AlgorithmInterface<Weight>::getArrayState(_grid),
				algorithm::AlgorithmInterface<Weight>::getArrayInterpretation(
						_grid),
				&algorithm::AlgorithmInterface<Weight>::getStateSize(_grid),
				&_current_rate), _current_time(0), _current_rate(
				0) {
	Embed();
}

template<class Weight>
void PopulationAlgorithm_<Weight>::Embed() {
	// at this point the local grid is big enough to accomodate expansion.

	// now the InitialDensity prescription together with V_min and the user-specified
	// number of bins are used to create an initial grid, which has a size given
	// by the initial number of bins
	InitializeAlgorithmGrid init;
	algorithm::AlgorithmGrid grid_initial = init.InitializeGrid(
			_parameter_specific.getNrGridInitial(),
			_parameter_specific.getVMin(), _parameter_population,
			_parameter_specific.getInitialDensity());
	// The initial grid now must be embedded in the local grid, which is generally larger.
	// In general the local grid of the PopulationAlgorithm is out of sync with the
	// valarrays maintained by the controller and it is only updated when a Report is due.
	// So this is done by the controller, since it maintans a local copy of the relevant
	// valarrays and needs insert the initial grid in the right sized valarray anyway.
	_controller_grid.EmbedGrid(_parameter_specific.getNrGridInitial(),
			this->getArrayState(grid_initial), // only Algorithms can unpack the valarrays
			this->getArrayInterpretation(grid_initial) // so they must be sent to the controller
					);
}

template<class Weight>
void PopulationAlgorithm_<Weight>::evolveNodeState(
		const std::vector<Rate>& nodeVector,
		const std::vector<Weight>& weightVector, Time time,
		const std::vector<NodeType>& typeVector) {

	_controller_grid.Evolve(time, &_current_time, &_current_rate, nodeVector,
			weightVector);

}

template<class Weight>
void PopulationAlgorithm_<Weight>::prepareEvolve(
		const std::vector<Rate>& nodeVector,
		const std::vector<Weight>& weightVector,
		const std::vector<NodeType>& typeVector) {
	_controller_grid.CollectExternalInput(nodeVector, weightVector, typeVector);

}

template<class Weight>
PopulationAlgorithm_<Weight>::~PopulationAlgorithm_() {
}

template<class Weight>
void PopulationAlgorithm_<Weight>::configure(
		const SimulationRunParameter& par_run) {

	// The grid controller is configured with each new configuration of the network.
	_controller_grid.Configure(par_run);

}



template<class Weight>
algorithm::AlgorithmGrid PopulationAlgorithm_<Weight>::getGrid() const {
	return _grid;
}

template<class Weight>
Rate PopulationAlgorithm_<Weight>::getCurrentRate() const {
  assert( _current_rate >= 0.0 );
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
