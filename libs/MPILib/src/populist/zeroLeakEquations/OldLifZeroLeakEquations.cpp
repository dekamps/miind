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
#include <MPILib/include/populist/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/populist/AbstractRateComputation.hpp>
#include <MPILib/include/populist/CirculantSolver.hpp>
#include <MPILib/include/populist/zeroLeakEquations/OldLifZeroLeakEquations.hpp>
#include <MPILib/include/populist/NonCirculantSolver.hpp>
#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>

using NumtoolsLib::IsApproximatelyEqualTo;

namespace MPILib {
namespace populist {

OldLIFZeroLeakEquations::OldLIFZeroLeakEquations(Number& n_bins,
		valarray<Potential>& array_state, Potential& check_sum,
		SpecialBins& bins,
		parameters::PopulationParameter& par_pop,//!< reference to the PopulationParameter
		parameters::PopulistSpecificParameter& par_spec,//!< reference to the PopulistSpecificParameter
		Potential& delta_v,		//!< reference to the current scale variable
		const AbstractCirculantSolver& circ,
		const AbstractNonCirculantSolver& noncirc) :
		LIFZeroLeakEquations(n_bins, array_state, check_sum, bins, par_pop,
				par_spec, delta_v, circ, noncirc), _time_current(0), _p_n_bins(
				&n_bins), _p_array_state(&array_state), _p_check_sum(
				&check_sum), _convertor( VALUE_REF_INIT
		bins, par_pop, par_spec, delta_v, n_bins), _p_solver_circulant(
				circ.Clone()), _p_solver_non_circulant(noncirc.Clone()) {
	this->SetInputParameter(_convertor.getSolverParameter());
}

void OldLIFZeroLeakEquations::Apply(Time time) {
	_time_current += time;
	assert(
			IsApproximatelyEqualTo(_p_array_state->sum()/(*_p_check_sum), 1.0, RELATIVE_LEAKAGE_PRECISION));

	ApplyZeroLeakEquationsAlphaInhibitory(time);
	ApplyZeroLeakEquationsAlphaExcitatory(time);

	assert(
			IsApproximatelyEqualTo( _p_array_state->sum()/(*_p_check_sum), 1.0, RELATIVE_LEAKAGE_PRECISION ));
}

void OldLIFZeroLeakEquations::ApplyZeroLeakEquationsAlphaExcitatory(Time time) {
	parameters::InputParameterSet& input_set = _convertor.getSolverParameter();

	double tau_e = input_set._rate_exc * time;
	double alpha_e = input_set._alpha_exc;

	// added to ignore zero input, which is legitimate (MdK: 11/06/2010)
	if (input_set._rate_exc == 0 && input_set._H_exc == 0)
		return;

#ifdef _INVESTIGATE_ALGORITHM
	ReportValue val;
	val._time = _time_current;
	CirculantSolver* p_solver = dynamic_cast<CirculantSolver*>(_p_solver_circulant.get());
	if (p_solver == 0)
	throw PopulistException("Can not downcast to CirculantSolver");
#endif

	assert(alpha_e <= 1.0 && alpha_e >= 0.0);
	if (input_set._n_circ_exc != 0 && tau_e > 0) {
		_p_solver_circulant->Execute(*_p_n_bins, (1 - alpha_e) * tau_e);
		_p_solver_non_circulant->ExecuteExcitatory(*_p_n_bins,
				(1 - alpha_e) * tau_e);
		_p_solver_circulant->AddCirculantToState(
				Bins()._index_current_reset_bin);
		// if alpha_e is close to zero, this is all that's necessary but if not then:
		// (this is always the case within the diffusion limit
#ifdef _INVESTIGATE_ALGORITHM
		val._value = input_set._rate_exc*p_solver->Flux(*_p_n_bins, tau_e);
		val._name_quantity = "alpha_e_pos_small";
		_p_convertor->Values().push_back(val);
#endif
		if (alpha_e > ALPHA_LIMIT) {
			input_set._H_exc++;
			_convertor.UpdateRestInputParameters();
			_p_solver_circulant->Execute(*_p_n_bins, alpha_e * tau_e);
			_p_solver_non_circulant->ExecuteExcitatory(*_p_n_bins,
					alpha_e * tau_e);
			_p_solver_circulant->AddCirculantToState(
					Bins()._index_current_reset_bin);
#ifdef _INVESTIGATE_ALGORITHM
			val._value = input_set._rate_exc*p_solver->Flux(*_p_n_bins, tau_e);
			val._name_quantity = "alpha_e_pos_large";
			_convertor.Values().push_back(val);
#endif
		}
	}
}

void OldLIFZeroLeakEquations::ApplyZeroLeakEquationsAlphaInhibitory(Time time) {
	parameters::InputParameterSet& input_set = _convertor.getSolverParameter();

	// added to ignore zero input, which is legitimate (MdK: 11/06/2010)
	if (input_set._rate_inh == 0 && input_set._H_inh == 0)
		return;

	double tau_i = input_set._rate_inh * time;
	double alpha_i = input_set._alpha_inh;

	assert(alpha_i <= 1.0 && alpha_i >= 0.0);
	if (input_set._rate_inh > 0 && tau_i > 0) {
		_p_solver_non_circulant->ExecuteInhibitory(*_p_n_bins,
				(1 - alpha_i) * tau_i);
		// alpha_i is close to zero, this is all that's necessary but if not then:
		// (this is always the case within the diffusion limit
		if (alpha_i > ALPHA_LIMIT) {
			input_set._H_inh++;
			_convertor.UpdateRestInputParameters();
			_p_solver_non_circulant->ExecuteInhibitory(*_p_n_bins,
					alpha_i * tau_i);
		}
	}
}

void OldLIFZeroLeakEquations::Configure(void* p_par	// irrelevant for LIFZeroLeakequations
		) {
	_convertor.Configure(this->ArrayState());
	parameters::InputParameterSet& input_set = _convertor.getSolverParameter();

	_p_solver_circulant->Configure(_p_array_state, input_set);

	_p_solver_non_circulant->Configure(*_p_array_state, input_set);

	_p_rate_calc = auto_ptr<AbstractRateComputation>(
			this->ParSpec().getRateComputation().Clone());

	_p_rate_calc->Configure(*_p_array_state, input_set, _convertor.getParPop(),
			_convertor.getIndexReversalBin());
}

Rate OldLIFZeroLeakEquations::CalculateRate() const {
	return _p_rate_calc->CalculateRate(*_p_n_bins);
}

void OldLIFZeroLeakEquations::RecalculateSolverParameters() {
	_convertor.RecalculateSolverParameters();
}

void OldLIFZeroLeakEquations::SortConnectionvector(
		const std::vector<Rate>& nodeVector,
		const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
		const std::vector<NodeType>& typeVector) {
	_convertor.SortConnectionvector(nodeVector, weightVector, typeVector);
}

void OldLIFZeroLeakEquations::AdaptParameters() {
	_convertor.AdaptParameters();
}

} /* namespace populist */
} /* namespace MPILib */
