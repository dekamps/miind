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
#ifndef _CODE_LIBS_POPULISTLIB_GEOMALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_GEOMALGORITHMCODE_INCLUDE_GUARD

#include "GeomAlgorithm.hpp"
#include "MasterFactory.hpp"

namespace GeomLib {

    template <class WeightValue>
	GeomAlgorithm<WeightValue>::GeomAlgorithm
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

	template <class WeightValue>
	GeomAlgorithm<WeightValue>::GeomAlgorithm
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

	template <class WeightValue>
	GeomAlgorithm<WeightValue>::~GeomAlgorithm()
	{
	}

	template <class WeightValue>
	GeomAlgorithm<WeightValue>* GeomAlgorithm<WeightValue>::clone() const
	{
	  return new GeomAlgorithm<WeightValue>(*this);
	}

	template<class WeightValue>
	void GeomAlgorithm<WeightValue>::configure(const SimulationRunParameter& par_run)
	{
		_t_cur		= par_run.getTBegin();
		_t_report	= par_run.getTReport();
	}

	template<class WeightValue>
	void GeomAlgorithm<WeightValue>::prepareEvolve
	(
		const std::vector<Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector,
		const std::vector<MPILib::NodeType>& typeVector
	)
	{
		_p_zl->sortConnectionVector(nodeVector,weightVector,typeVector);
	}

	template <class WeightValue>
	void GeomAlgorithm<WeightValue>::evolveNodeState
	(
		const std::vector<Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector,
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
		    _p_zl->apply(_t_step);

		}
		_t_cur = _p_system->CurrentTime();


		// previously, a report was prepared here at report time. That is
		// unnecessary. This can be handled in the Grid method.

	}

	template <class WeightValue>
	Time GeomAlgorithm<WeightValue>::getCurrentTime() const
	{
		return _t_cur;
	}

	template <class WeightValue>
	bool GeomAlgorithm<WeightValue>::IsReportDue() const
	{
		if (_n_report*_t_report < _t_cur){
			++_n_report;
			return true;
		}
		else 
			return false;
			
	}
/*
	template <class WeightValue>
	NodeState GeomAlgorithm<WeightValue>::State() const
	{
		return NodeState(vector<double>(0));
	}
*/
	template <class WeightValue>
	AlgorithmGrid GeomAlgorithm<WeightValue>::getGrid() const
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

	template <class WeightValue>
	Rate GeomAlgorithm<WeightValue>::getCurrentRate() const
	{
		return _p_system->CurrentRate() + _p_zl->getTransitionRate();
	}
}

#endif // include guard

