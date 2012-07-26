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
#include <cassert>
#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/populist/circulantSolvers/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/populist/zeroLeakEquations/SingleInputZeroLeakEquations.hpp>
#include <MPILib/include/utilities/Exception.hpp>

using NumtoolsLib::IsApproximatelyEqualTo;

namespace MPILib {
namespace populist {

SingleInputZeroLeakEquations::SingleInputZeroLeakEquations(Number& n_bins,
		valarray<Potential>& array_state, Potential& check_sum,
		SpecialBins& bins,
		parameters::PopulationParameter& par_pop,//!< reference to the PopulationParameter
		parameters::PopulistSpecificParameter& par_spec,//!< reference to the PopulistSpecificParameter
		Potential& delta_v,		//!< reference to the current scale variable
		const circulantSolvers::AbstractCirculantSolver& circ,
		const nonCirculantSolvers::AbstractNonCirculantSolver& noncirc) :
		LIFZeroLeakEquations(n_bins, array_state, check_sum, bins, par_pop,
				par_spec, delta_v, circ, noncirc) {
	this->SetMode(FLOATING_POINT, *_p_solver_circulant);
	this->SetMode(FLOATING_POINT, *_p_solver_non_circulant);
	this->SetInputParameter(_convertor.getSolverParameter());
}

void SingleInputZeroLeakEquations::Apply(Time time) {
	Time t_evolve = time * Set()._rate_exc;
	if (_p_solver_circulant->BeforeNonCirculant()) {
		_p_solver_circulant->Execute(*_p_n_bins, t_evolve, _time_current);
		_p_solver_non_circulant->ExecuteExcitatory(*_p_n_bins, t_evolve);
		_p_solver_circulant->AddCirculantToState(
				Bins()._index_current_reset_bin);
	} else {
		_p_solver_non_circulant->ExecuteExcitatory(*_p_n_bins, t_evolve);
		_p_solver_circulant->Execute(*_p_n_bins, t_evolve, _time_current);
		_p_solver_circulant->AddCirculantToState(
				Bins()._index_current_reset_bin);
	}

	_time_current += time;

	// This assert may be triggered if the algorithm is run for more  0.5 s.
	// CirculantAlgorithm slowly builds up a proability mismatch over longer times
	assert(
			IsApproximatelyEqualTo(_p_array_state->sum() + _p_solver_circulant->RefractiveProbability(),1.0,1e-7));
}

} /* namespace populist */
} /* namespace MPILib */
