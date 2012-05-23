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
#ifndef _CODE_LIBS_POPULISTLIB_ONEDMALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_ONEDMALGORITHMCODE_INCLUDE_GUARD

#include "OneDMAlgorithm.h"
#include "InitializeAlgorithmGrid.h"

namespace PopulistLib {

	template <class Weight>
	OneDMAlgorithm<Weight>::OneDMAlgorithm(const OneDMParameter& par):
	AbstractAlgorithm<Weight>	(0),
	_parameter_onedm			(par),
	_grid						(AlgorithmGrid(par._par_spec.MaxNumGridPoints())),
	_controller_grid
	(
		VALUE_MEMBER_ARG		
		ToPopulationParameter(par),
		_parameter_onedm._par_spec,
		AbstractAlgorithm<Weight>::ArrayState         (_grid),
		AbstractAlgorithm<Weight>::ArrayInterpretation(_grid),
		&AbstractAlgorithm<Weight>::StateSize(_grid),
		&_current_rate,
		&_stream_log
	)
	{
	}

	template <class Weight>
	OneDMAlgorithm<Weight>::OneDMAlgorithm
	(
		const OneDMAlgorithm<Weight>& rhs
	):
	AbstractAlgorithm<Weight>(rhs),
	_parameter_onedm	(rhs._parameter_onedm),
	_grid				(rhs._grid),
	_controller_grid
	(
#ifdef _INVESTIGATE_ALGORITHM
		_vec_value,
#endif
		ToPopulationParameter(rhs._parameter_onedm),
		rhs._parameter_onedm._par_spec,
		AbstractAlgorithm<Weight>::ArrayState         (_grid),
		AbstractAlgorithm<Weight>::ArrayInterpretation(_grid),
		&AbstractAlgorithm<Weight>::StateSize(_grid),
		&_current_rate,
		&_stream_log
	)
	{
	}

	template <class Weight>
	OneDMAlgorithm<Weight>::~OneDMAlgorithm()
	{
	}

	template <class Weight>
	bool OneDMAlgorithm<Weight>::Configure(const SimulationRunParameter& par_run) 
	{

	_grid = AlgorithmGrid(_parameter_onedm._par_spec.MaxNumGridPoints());

		InitializeAlgorithmGrid init;

		// One difference with respect to PopulationAlgorithm:
		// the moving frame parameters are defined in terms of g, not v.
		// So a conversion has to take place for the benefit of the PopulationGridController
		// which is unaware that the relevant state is g.

		PopulationParameter par_pop;
		par_pop._tau   = _parameter_onedm._par_adapt._t_adaptation;
		par_pop._theta = _parameter_onedm._par_adapt._g_max;

		// now the InitialDensity prescription together with V_min and the user-specified
		// number of bins are used to create an initial grid, which has a size given
		// by the initial number of bins
		AlgorithmGrid grid_initial = 
			init.InitializeGrid
			(
				_parameter_onedm._par_spec.NrGridInitial(),
				_parameter_onedm._par_spec.VMin(),
				par_pop,
				_parameter_onedm._par_spec.InitialDensity()
			);

		// the grid controller is configured with each new configuration of the network
		_controller_grid.Configure
		(
			par_run
		);

		// The initial grid now must be embedded in the local grid, which is generally larger.
		// In general the local grid of the PopulationAlgorithm is out of sync with the 
		// valarrays maintained by the controller and it is only updated when a Report is due.
		// So this is done by the controller, since it maintans a local copy of the relevant
		// valarrays and needs insert the initial grid in the right sized valarray anyway.
		_controller_grid.EmbedGrid
		(
			_parameter_onedm._par_spec.NrGridInitial(),
			this->ArrayState			(grid_initial), // only Algorithms can unpack the valarrays
			this->ArrayInterpretation	(grid_initial),  // so they must be sent to the controller
			&_parameter_onedm	// here the OneDMParameter is passed into the ZeroleakEquations, without opulationGridController needing to know its type
		);

		// write information in the log file with regard to algorithm settings
		WriteConfigurationToLog();

		return true;
	}
	template <class Weight>
	Rate OneDMAlgorithm<Weight>::CurrentRate() const 
	{
		return _current_rate;
	}

	template <class Weight>
	Time OneDMAlgorithm<Weight>::CurrentTime() const 
	{
		return _current_time;
	}

	template <class Weight>
	PopulationParameter OneDMAlgorithm<Weight>::ToPopulationParameter
	(
		const OneDMParameter& par_one
	)
	{
		PopulationParameter par_pop;
		par_pop._tau    = par_one._par_adapt._t_adaptation;
		par_pop._theta  = par_one._par_adapt._g_max;

		return par_pop;
	}

	template <class Weight>
	bool OneDMAlgorithm<Weight>::EvolveNodeState
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end,
		Time time
	)
	{
		bool b_return =  
			_controller_grid.Evolve
			(
				time,
				&_current_time,
				&_current_rate,
				iter_begin,
				iter_end
			);	

		return b_return;
	}

	template <class Weight>
	AlgorithmGrid OneDMAlgorithm<Weight>::Grid() const 
	{
		return _grid;
	}

	template <class Weight>
	NodeState OneDMAlgorithm<Weight>::State() const 
	{
		vector<double> vector_return(0);
		return NodeState(vector_return);
	}

	template <class Weight>
	bool OneDMAlgorithm<Weight>::Dump(ostream& s) const
	{
		return false;
	}

		template <class Weight>
	string OneDMAlgorithm<Weight>::LogString() const
	{
		string string_return =_stream_log.str();
		_stream_log.clear();
		return string_return;
	}

	template <class Weight>
	void OneDMAlgorithm<Weight>::WriteConfigurationToLog()
	{
	}
}
#endif //include guard