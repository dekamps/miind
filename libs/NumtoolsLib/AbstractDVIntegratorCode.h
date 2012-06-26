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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NUMTOOLSLIB_NODE_ABSTRACTDVINTEGRATORCODE_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_NODE_ABSTRACTDVINTEGRATORCODE_INCLUDE_GUARD

#include "AbstractDVIntegrator.h"
#include "DVIntegratorException.h"
#include "LocalDefinitions.h"


namespace NumtoolsLib {

	template <class ParameterObject>
	AbstractDVIntegrator<ParameterObject>::AbstractDVIntegrator
	(
		Time						t_begin,
		TimeStep					t_step,
		const Precision&			precision,
		const gsl_odeiv_step_type*	p_step_algorithm,
		double*						p_state,
		Number						state_size,
		Function					function,
		Derivative					derivative,
		Number						max_number_iterations
	):
	_time_begin			(t_begin),
	_time_current		(t_begin),
	_step				(t_step),
	_precision			(precision),
	_gsl_objects
	(
		p_step_algorithm,
		state_size,
		precision
	),
	_p_state				(p_state),
	_state_size				(state_size),
	_function				(function),
	_derivative				(derivative),
	_number_iterations		(0),
	_max_number_iterations	(max_number_iterations)
	{
		assert( _state_size > 0 );
		_gsl_objects._system.dimension = _state_size;
		_gsl_objects._system.function  = _function;
		_gsl_objects._system.jacobian  = _derivative;
		_gsl_objects._system.params    = &_parameter_space;
	}

	template <class ParameterObject>
	AbstractDVIntegrator<ParameterObject>::AbstractDVIntegrator
	(
		const AbstractDVIntegrator<ParameterObject>& rhs
	):
	UtilLib::Streamable(),
	_time_begin				(rhs._time_begin),
	_time_current			(rhs._time_current),
	_step					(rhs._step),
	_precision				(rhs._precision),
	_gsl_objects			(rhs._gsl_objects._p_step_algorithm,rhs._state_size,rhs._precision),
	_state_size				(rhs._state_size),
	_function				(rhs._function),
	_derivative				(rhs._derivative),
	_number_iterations		(rhs._number_iterations),
	_max_number_iterations	(rhs._max_number_iterations)
	{
		assert( _state_size > 0 );

		
		_gsl_objects._system.dimension = _state_size;
		_gsl_objects._system.function  = _function;
		_gsl_objects._system.jacobian  = _derivative;
		_gsl_objects._system.params    = &_parameter_space;

	}


	template <class ParameterObject>
	AbstractDVIntegrator<ParameterObject>::~AbstractDVIntegrator()
	{
	}

	template <class ParameterObject>
	Time AbstractDVIntegrator<ParameterObject>::Evolve(Time t_end)
	{
		int status = 
			gsl_odeiv_evolve_apply
			(
				_gsl_objects._p_evolver, 
				_gsl_objects._p_controller, 
				_gsl_objects._p_step,
				&_gsl_objects._system, 
				&_time_current, 
				t_end,
				&_step,
				_p_state
			);

			++_number_iterations;

			if ( _number_iterations > _max_number_iterations )
				throw DVIntegratorException
				(
					NUMBER_ITERATIONS_EXCEEDED,
					STRING_MAX_ITERATIONS
				);

			if (status != GSL_SUCCESS)
				 throw 
					DVIntegratorException
					(
						INTEGRATION_FAILED,
						string(gsl_strerror(status))
					);

			return _time_current;
	}
}
#endif 
