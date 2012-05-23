// Copyright (c) 2005 - 2010 Marc de Kamps
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
#include <cassert>
#include "../NumtoolsLib/NumtoolsLib.h"
#include "AbstractCirculantSolver.h"
#include "AbstractNonCirculantSolver.h"
#include "CirculantSolver.h"
#include "LocalDefinitions.h"
#include "DiffusionZeroLeakEquations.h"
#include "NonCirculantSolver.h"
#include "PopulistException.h"

using namespace NumtoolsLib;
using namespace PopulistLib;

DiffusionZeroLeakEquations::DiffusionZeroLeakEquations
(
	Number&						n_bins,		
	valarray<Potential>&		array_state,
	Potential&					check_sum,
	ConvertMuSigmaToH&			convertor
):
_time_current(0),
_p_n_bins(&n_bins),
_p_array_state(&array_state),
_p_check_sum(&check_sum),
_p_convertor(&convertor),
_p_solver_circulant(0),
_p_solver_non_circulant(0)
{
}

void DiffusionZeroLeakEquations::Apply(PopulistLib::Time time)
{
	_time_current += time;
	assert( IsApproximatelyEqualTo(_p_array_state->sum()/(*_p_check_sum), 1.0, RELATIVE_LEAKAGE_PRECISION) );

	ApplyZeroLeakEquationsAlphaExcitatory(time);

	assert ( IsApproximatelyEqualTo( _p_array_state->sum()/(*_p_check_sum), 1.0, RELATIVE_LEAKAGE_PRECISION ) );
}

void DiffusionZeroLeakEquations::ApplyZeroLeakEquationsAlphaExcitatory(PopulistLib::Time time)
{
	InputParameterSet& input_set = _p_convertor->SolverParameter();

	if (input_set._rate_exc == 0 && input_set._H_exc == 0 )
		return;

	double tau_e    = input_set._rate_exc*time;
	double tau_i    = input_set._rate_inh*time;
	double tau		= tau_e - tau_i;

	if ( tau >= 0 ){
		_p_solver_non_circulant->ExecuteExcitatory(*_p_n_bins,tau);
		_p_solver_circulant->Execute(*_p_n_bins,tau);
		_p_solver_circulant->AddCirculantToState();
	}
	else {
		_p_solver_non_circulant->ExecuteInhibitory(*_p_n_bins,tau);
	}


}

void DiffusionZeroLeakEquations::ApplyZeroLeakEquationsAlphaInhibitory(PopulistLib::Time time)
{
	InputParameterSet& input_set = _p_convertor->SolverParameter();

	// added to ignore zero input, which is legitimate (MdK: 11/06/2010)
	if (input_set._rate_inh == 0 && input_set._H_inh == 0 )
		return;

	double tau_i   = input_set._rate_inh*time;
	double alpha_i = input_set._alpha_inh;

	assert (alpha_i <= 0.5 && alpha_i >= -0.5);
	if ( input_set._rate_inh > 0 && tau_i > 0 )
	{
		if ( alpha_i > 0 )
		{
			_p_solver_non_circulant->ExecuteInhibitory(*_p_n_bins,(1-alpha_i)*tau_i);
			// alpha_i is close to zero, this is all that's necessary but if not then:
			// (this is always the case within the diffusion limit
			if ( alpha_i > ALPHA_LIMIT )
			{
				input_set._H_inh++;
				_p_convertor->UpdateRestInputParameters();
				_p_solver_non_circulant->ExecuteInhibitory(*_p_n_bins,alpha_i*tau_i);
			}
		}
		else
		{
			_p_solver_non_circulant->ExecuteInhibitory(*_p_n_bins,(1+alpha_i)*tau_i);
			// alpha_i is close to zero, this is all that's necessary but if not then:
			// (this is always the case within the diffusion limit)
			if ( alpha_i > ALPHA_LIMIT )
			{
				input_set._H_inh--;
				_p_convertor->UpdateRestInputParameters();
				_p_solver_non_circulant->ExecuteInhibitory(*_p_n_bins,alpha_i*tau_i);
			}
		}
	}
}

void DiffusionZeroLeakEquations::Configure
(
	const PopulistSpecificParameter& par_spec
)
{
	InputParameterSet& input_set = _p_convertor->SolverParameter();

	if ( par_spec.CirculantSolver() == 0 )
		_p_solver_circulant = auto_ptr<AbstractCirculantSolver>(new CirculantSolver);
	else
		_p_solver_circulant = auto_ptr<AbstractCirculantSolver>( par_spec.CirculantSolver()->Clone() );

	if ( par_spec.NonCirculantSolver() == 0)
		_p_solver_non_circulant = auto_ptr<AbstractNonCirculantSolver>(new NonCirculantSolver);
	else
		_p_solver_non_circulant = auto_ptr<AbstractNonCirculantSolver>( par_spec.NonCirculantSolver()->Clone() );

	_p_solver_circulant->Configure
	(
		_p_array_state,
		*_p_convertor
	);


	_p_solver_non_circulant->Configure
	(
		*_p_array_state,
		input_set
	);

	_p_rate_calc = auto_ptr<AbstractRateComputation>(par_spec.RateComputation().Clone());

	_p_rate_calc->Configure
	(
		*_p_array_state,
		input_set,
		_p_convertor->ParPop(), 
		_p_convertor->IndexReversalBin()
	);
}

Rate DiffusionZeroLeakEquations::CalculateRate() const
{
	return _p_rate_calc->CalculateRate(*_p_n_bins);
}