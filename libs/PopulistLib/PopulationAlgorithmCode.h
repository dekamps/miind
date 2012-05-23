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
#ifndef _CODE_LIBS_POPULISTLIB_POPULATIONALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_POPULATIONALGORITHMCODE_INCLUDE_GUARD

#include "PopulationAlgorithm.h"
#include "ConnectionSquaredProduct.h"
#include "PopulationAlgorithm.h"
#include "InitializeAlgorithmGrid.h"
#include "LocalDefinitions.h"
#include "PopulistParameter.h"
#include "PopulistException.h"

namespace PopulistLib {

	template <class Weight>
	PopulationAlgorithm_<Weight>::PopulationAlgorithm_
	(
		const PopulistParameter& par_populist
	):
	AbstractAlgorithm<PopulationConnection>(0),
	_parameter_population(par_populist._par_pop),
	_parameter_specific(par_populist._par_spec),
	_grid(AlgorithmGrid(_parameter_specific.MaxNumGridPoints())),
	_controller_grid
	(
#ifdef _INVESTIGATE_ALGORITHM
		_vec_value,
#endif
		_parameter_population,
		_parameter_specific,
		AbstractAlgorithm<Weight>::ArrayState			(_grid),
		AbstractAlgorithm<Weight>::ArrayInterpretation	(_grid),
		&AbstractAlgorithm<Weight>::StateSize			(_grid),
		&_current_rate,
		&_stream_log
	),
	_current_time(0),
	_current_rate(0)
	{
		Embed();
	}

	template <class Weight>
	PopulationAlgorithm_<Weight>::PopulationAlgorithm_
	(
		istream& s
	):
	AbstractAlgorithm<PopulationConnection>(0),
	_parameter_population(ParPopFromStream(s)),
	_parameter_specific(ParSpecFromStream(s)),
	_grid(AlgorithmGrid(_parameter_specific.MaxNumGridPoints())),
	_controller_grid
	(
#ifdef _INVESTIGATE_ALGORITHM
		_vec_value,
#endif
		_parameter_population,
		_parameter_specific,
		AbstractAlgorithm<Weight>::ArrayState			(_grid),
		AbstractAlgorithm<Weight>::ArrayInterpretation	(_grid),
		&AbstractAlgorithm<Weight>::StateSize			(_grid),
		&_current_rate,
		&_stream_log
	),
	_current_time(0),
	_current_rate(0)
	{
		_stream_log << "Running with ZeroLeakEquations: "	<< _parameter_specific.ZeroLeakName()		<< "\n";
		_stream_log << "Running with Circulant: "			<< _parameter_specific.CirculantName()		<< "\n";
		_stream_log << "Running with NonCirculant"			<< _parameter_specific.NonCirculantName()	<< "\n";
		Embed();
		StripFooter(s);
	}

	template <class Weight>
	PopulationParameter	PopulationAlgorithm_<Weight>::ParPopFromStream(istream& s)
	{
		StripHeader(s);
		PopulationParameter par;
		par.FromStream(s);
		return par;
	}

	template <class Weight>
	PopulistSpecificParameter PopulationAlgorithm_<Weight>::ParSpecFromStream(istream& s)
	{
		PopulistSpecificParameter par;
		par.FromStream(s);
		return par;
	}

	template <class Weight>
	void PopulationAlgorithm_<Weight>::StripHeader(istream& s)
	{
		string dummy;

		s >> dummy;
		// Either an AbstractAlgorithm tag and then a PopulationAlgorithm tag, or just a PopulationAlgorithm tag when the stream already has been processed by a builder

		if ( this->IsAbstractAlgorithmTag(dummy) ){
			s >> dummy;
			s >> dummy;
		}

		ostringstream str;
		str << dummy << ">";
		if ( str.str() != this->Tag() )
			throw PopulistException("PopulationAlgorithm tag expected");

		getline(s,dummy);

		string name_alg;
		if (! this->StripNameFromTag(&name_alg, dummy) )
			throw PopulistException("PopulationAlgorithm tag expected");
		this->SetName(name_alg);
	}

	template <class Weight>
	void PopulationAlgorithm_<Weight>::StripFooter(istream& s)
	{
		string dummy;

		s >> dummy;

		if ( dummy != this->ToEndTag(this->Tag() ) )
			throw PopulistException("PopulationAlgorithm end tag expected");

		// absorb the AbstractAlgorithm tag
		s >> dummy;

	}

	template <class Weight>
	PopulationAlgorithm_<Weight>::PopulationAlgorithm_
	(
		const PopulationAlgorithm_<Weight>& algorithm
	):
	AbstractAlgorithm<PopulationConnection>(algorithm),
	_parameter_population(algorithm._parameter_population),
	_parameter_specific(algorithm._parameter_specific),
	_grid(AlgorithmGrid(_parameter_specific.MaxNumGridPoints())),
	_controller_grid
	(
#ifdef _INVESTIGATE_ALGORITHM
		_vec_value,
#endif
		_parameter_population,
		_parameter_specific,
		AbstractAlgorithm<Weight>::ArrayState         (_grid),
		AbstractAlgorithm<Weight>::ArrayInterpretation(_grid),
		&AbstractAlgorithm<Weight>::StateSize(_grid),
		&_current_rate,
		&_stream_log
	),
	_current_time(0),
	_current_rate(0)
	{
		Embed();
	}


	template <class Weight>
	void PopulationAlgorithm_<Weight>::Embed
	(
	)
	{
		// at this point the local grid is big enough to accomodate expansion.

		// now the InitialDensity prescription together with V_min and the user-specified
		// number of bins are used to create an initial grid, which has a size given
		// by the initial number of bins
		InitializeAlgorithmGrid init;
		AlgorithmGrid grid_initial = 
			init.InitializeGrid
			(
				_parameter_specific.NrGridInitial(),
				_parameter_specific.VMin(),
				_parameter_population,
				_parameter_specific.InitialDensity()
			);
		// The initial grid now must be embedded in the local grid, which is generally larger.
		// In general the local grid of the PopulationAlgorithm is out of sync with the 
		// valarrays maintained by the controller and it is only updated when a Report is due.
		// So this is done by the controller, since it maintans a local copy of the relevant
		// valarrays and needs insert the initial grid in the right sized valarray anyway.
		_controller_grid.EmbedGrid
		(
			_parameter_specific.NrGridInitial(),
			this->ArrayState			(grid_initial), // only Algorithms can unpack the valarrays
			this->ArrayInterpretation	(grid_initial)  // so they must be sent to the controller
		);
	}

	template <class Weight>
	bool PopulationAlgorithm_<Weight>::EvolveNodeState
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
	bool PopulationAlgorithm_<Weight>::CollectExternalInput
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end
	)
	{
		bool b_return =
			_controller_grid.CollectExternalInput
			(
				iter_begin,
				iter_end
			);

		return b_return;
	}

	template <class Weight>
	PopulationAlgorithm_<Weight>::~PopulationAlgorithm_() 
	{
	}

	template <class Weight>
	bool PopulationAlgorithm_<Weight>::Configure(const SimulationRunParameter& par_run) 
	{

	

		// The grid controller is configured with each new configuration of the network.
		_controller_grid.Configure
		(
			par_run
		);


		// write information in the log file with regard to algorithm settings
		WriteConfigurationToLog();

		return true;
	}

	template <class Weight>
	bool PopulationAlgorithm_<Weight>::Dump(ostream& s) const 
	{
		return true;
	}

	template <class Weight>
	AlgorithmGrid PopulationAlgorithm_<Weight>::Grid() const 
	{
		return _grid;
	}

	template <class Weight>
	NodeState PopulationAlgorithm_<Weight>::State() const 
	{
		vector<double> vector_return(0);
		return NodeState(vector_return);
	}

	template <class Weight>
	Rate PopulationAlgorithm_<Weight>::CurrentRate() const 
	{
		return _current_rate;
	}

	template <class Weight>
	Time PopulationAlgorithm_<Weight>::CurrentTime() const 
	{
		return _current_time;
	}

	template <class Weight>
	string PopulationAlgorithm_<Weight>::LogString() const
	{
		string string_return =_stream_log.str();
		_stream_log.clear();
		return string_return;
	}

	template <class Weight>
	Potential PopulationAlgorithm_<Weight>::BinToCurrentPotential(Index index) const
	{
		return _controller_grid.BinToCurrentPotential(index);
	}

	template <class Weight>
	Index PopulationAlgorithm_<Weight>::CurrentPotentialToBin(Potential v) const
	{
		return _controller_grid.CurrentPotentialToBin(v);
	}
	
	template <class Weight>
	void PopulationAlgorithm_<Weight>::WriteConfigurationToLog()
	{
	}


	template <class Weight>
	string PopulationAlgorithm_<Weight>::Tag() const
	{
		return STR_POPALGORITHM_TAG;
	}

	template <class Weight>
	bool PopulationAlgorithm_<Weight>::ToStream(ostream& s) const
	{
		this->AbstractAlgorithm<Weight>::ApplyBaseClassHeader(s,"PopulationAlgorithm");

		s << this->InsertNameInTag(this->Tag(),this->GetName()) << "\n";
		_parameter_population.ToStream(s);
		_parameter_specific.ToStream(s);
		s << this->ToEndTag(this->Tag()) << "\n";
		this->AbstractAlgorithm<Weight>::ApplyBaseClassFooter(s);

		return true;
	}


	template <class Weight>
	bool PopulationAlgorithm_<Weight>::FromStream(istream& s)
	{
		return false;
	}

} // end of namespace

#endif // include guard
