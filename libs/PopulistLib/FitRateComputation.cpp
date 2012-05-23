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
#include "FitRateComputation.h"
#include "LocalDefinitions.h"

using namespace PopulistLib;

const int DEGREE = 3;

FitRateComputation::FitRateComputation():
_X(0),
_cov(0),
_c(0),
_p_work(0)
{
}

FitRateComputation::~FitRateComputation()
{
	Free();
}


void FitRateComputation::Free()
{
	if (_X)
		gsl_matrix_free(_X);
	if (_cov)
		gsl_matrix_free(_cov);
	if (_c)
		gsl_vector_free(_c);

//	if (_p_work)
//		gsl_multifit_linear_free(_p_work); 
}

void FitRateComputation::Alloc()
{
	_X = gsl_matrix_alloc   (_n_points,  DEGREE + 1);
	_cov = gsl_matrix_alloc (DEGREE + 1, DEGREE + 1);

	_c = gsl_vector_alloc (DEGREE + 1);
	  
}

void FitRateComputation::Configure
(
	valarray<Density>&			array_state,
 	const InputParameterSet&	input_set,	// current input to population
	const PopulationParameter&	par_pol,	// neuron parameters
	Index						index_reversal	// index reversal bin
)
{	
	AbstractRateComputation::Configure
	(
		array_state,
		input_set,
		par_pol,
		index_reversal
	);

	_n_points = static_cast<Number>(array_state.size());
	_array_interpretation.resize(_n_points);

	Free();
	Alloc();
}

FitRateComputation* FitRateComputation::Clone() const
{
	// type transfer, no state transfer
	return new FitRateComputation;
}

void FitRateComputation::SetUpFittingMatrices()
{
	_y.owner  = 0;  // use data of the _state_array
	_y.block  = 0;	
	_y.data   = &(*_p_array_state)[_start_integration_area];
	_y.size   = _number_integration_area;
	_y.stride = 1;

	// restrict the size on the data matrix
	_X->size1 = _number_integration_area;

	// ditto for the workspace
//	_p_work->n = _number_integration_area;


	for (Number i = 0; i < _number_integration_area; i++)
	{
 
		// first matrix column
		double v_p = 1.0;
		gsl_matrix_set
		(
			_X, 
			i, 
			0, 
			v_p
		);

		// and the rest
		for ( int j = 1; j < DEGREE + 1; j++)
		{
			v_p *= _array_interpretation[_start_integration_area + i];
			gsl_matrix_set
			(
				_X, 
				i, 
				j, 
				v_p
			);
		}
 	}
}

Rate FitRateComputation::CalculateRate(Number n_bins)
{
	_n_bins = n_bins;

	Potential v_cutoff =  (1.0 - FIT_LOWER_BOUND);

	DefineRateArea(v_cutoff);

	SetUpFittingMatrices();

	double chisq;

	_p_work =	
		gsl_multifit_linear_alloc 
		(
			_number_integration_area, 
			DEGREE + 1
		);	
	
	gsl_multifit_linear
	(
		_X, 
		&_y, 
		_c, 
		_cov,
                &chisq, 
		_p_work
	);

	gsl_multifit_linear_free(_p_work);
	return ExtractRateFromFit();
}

bool FitRateComputation::IsSingleInput() const
{
	return ( _p_input_set->_rate_inh == 0 || _p_input_set->_rate_exc == 0) ? true : false;
}

Rate FitRateComputation::ExtractRateFromFit()
{
//	const valarray<double>& _array_state = *_p_array_state;
/*
//	double  a0 = gsl_vector_get(_c,0);
	double  a1 = gsl_vector_get(_c,1);
	double  a2 = gsl_vector_get(_c,2);
	double  a3 = gsl_vector_get(_c,3);
 
	double deriv       = a1 + 2.0*a2 + 3.0*a3;
	double deriv_prime = 2.0*a2 + 6.0*a3;

	double mu  = _p_input_set->_par_input._mu;
	double var = _p_input_set->_par_input._sigma;

	Potential DeltaV = (_par_population._theta - _par_population._V_reversal); 

	Rate rate_return;


	if (IsSingleInput()){

		rate_return =	(mu > 0) ? 
				(
					-var*var*deriv/(2.0*_par_population._tau*DeltaV*DeltaV*_delta_v_rel) +\
					var*var*var*var*deriv_prime/(mu*6.0*_par_population._tau*DeltaV*DeltaV*_delta_v_rel) 
				) : 0;
	}
	else {

		rate_return = -var*var*deriv/(2.0*_par_population._tau*DeltaV*DeltaV*_delta_v_rel);

		// after strong feedback, this may happen
		// TODO: decide on warning
		if (rate_return < 0 )
			rate_return = 0;

	}

	return rate_return;*/ return 0.0;
}
