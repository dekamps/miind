// Copyright (c) 2005 - 2014 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_QIFALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_QIFALGORITHMCODE_INCLUDE_GUARD

#include "QIFAlgorithm.h"
#include "MasterFactory.h"

namespace PopulistLib {

    template <class Weight>
	GeomAlgorithm<Weight>::GeomAlgorithm
	(
		const GeomParameter& par_geom
	):
	_par_geom(par_geom),
	_grid(0),
	_p_system(par_geom._p_sys_ode->Clone()),
    _b_zl(!par_geom._no_master_equation),
	_t_cur(0.0),
	_t_report(0.0),
	_n_report(0)
	{
		_grid = AlgorithmGrid(_p_system->NumberOfBins());
		_t_step = _p_system->TStep();

		MasterFactory fact;
		_p_zl = fact.Create
		  (
		   _par_geom._name_master,
		   *_p_system,
		   par_geom._par_diff,
		   par_geom._par_cur
		   );
	}

	template <class Weight>
	GeomAlgorithm<Weight>::GeomAlgorithm
	(
		const GeomAlgorithm& alg
	):
	_par_geom				(alg._par_geom),
	_grid					(alg._grid),
	_p_system				(alg._p_system->Clone()),
    _b_zl           		(alg._b_zl),
	_t_cur					(alg._t_cur),
    _t_step         		(alg._t_step),
	_t_report				(alg._t_report),
	_n_report				(alg._n_report)
	{
		_grid = AlgorithmGrid(_p_system->NumberOfBins());

		MasterFactory fact;
		_p_zl = fact.Create
		  (
		   _par_geom._name_master,
		   *_p_system,
		   _par_geom._par_diff,
		   _par_geom._par_cur
		  );
	}

	template <class Weight>
	GeomAlgorithm<Weight>::~GeomAlgorithm()
	{
	}

	template <class Weight>
	GeomAlgorithm<Weight>* GeomAlgorithm<Weight>::Clone() const
	{
	  return new GeomAlgorithm<Weight>(*this);
	}

	template <class Weight>
	string GeomAlgorithm<Weight>::LogString() const
	{
		return "";
	}

	template <class Weight>
	bool GeomAlgorithm<Weight>::Dump(ostream&) const
	{
		return true;
	}

	template<class Weight>
	bool GeomAlgorithm<Weight>::Configure(const SimulationRunParameter& par_run)
	{
		_t_cur		= par_run.TBegin();
		_t_report	= par_run.TReport();

		return true;
	}

	template <class Weight>
	bool GeomAlgorithm<Weight>::CollectExternalInput
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end
	)
	{
		_p_zl->SortConnectionvector(iter_begin,iter_end);
		return true;
	}

	template <class Weight>
	bool GeomAlgorithm<Weight>::EvolveNodeState
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end,
		Time t
	)
	{
	    double n = (t - _t_cur)/_t_step;
		Number n_steps = static_cast<Number>(ceil(n));
		if (n_steps == 0)
		  n_steps++;

		for (Index i = 0; i < n_steps; i++){
		  _p_system->Evolve(_t_step);
		  if (_b_zl)
		    _p_zl->Apply(_t_step);

		}
		_t_cur = _p_system->CurrentTime();

		// previously, a report was prepared here at report time. That is
		// unnecessary. This can be handled in the Grid method.


		return true;
	}

	template <class Weight>
	Time GeomAlgorithm<Weight>::CurrentTime() const
	{
		return _t_cur;
	}

	template <class Weight>
	bool GeomAlgorithm<Weight>::IsReportDue() const
	{
		if (_n_report*_t_report < _t_cur){
			++_n_report;
			return true;
		}
		else 
			return false;
			
	}

	template <class Weight>
	NodeState GeomAlgorithm<Weight>::State() const
	{
		return NodeState(vector<double>(0));
	}

	template <class Weight>
	AlgorithmGrid GeomAlgorithm<Weight>::Grid() const
	{
		Number N = _p_system->NumberOfBins();
		vector<double> array_interpretation(N);
		vector<double> array_state(N);
		_p_system->PrepareReport
				(
						&(array_interpretation[0]),
						&(array_state[0])
				);
		return AlgorithmGrid(array_state,array_interpretation);
	}

	template <class Weight>
	Rate GeomAlgorithm<Weight>::CurrentRate() const
	{
		return _p_system->CurrentRate() + _p_zl->TransitionRate();
	}
}

#endif // include guard

