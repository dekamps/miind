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
#include "GSLObjects.h"

using namespace NumtoolsLib;

GSLObjects::GSLObjects
(
	const gsl_odeiv_step_type* p_step_algorithm,	//<! choice for solver type
	Number state_size,								//<! state size
	const Precision& precision						//<! absolute and relative precision conform GSL documentation
):
_p_step_algorithm(p_step_algorithm),
_p_step
(
	gsl_odeiv_step_alloc
	(
		_p_step_algorithm, 
		state_size
	)
),
_p_controller
(
	gsl_odeiv_control_y_new
	(
		precision._eps_absolute,
		precision._eps_relative
	)
),
_p_evolver			
(
	gsl_odeiv_evolve_alloc 
	(
		state_size
	)
)
{
}

GSLObjects::~GSLObjects()
{
	gsl_odeiv_evolve_free	(_p_evolver);
	gsl_odeiv_control_free	(_p_controller);
	gsl_odeiv_step_free		(_p_step);
}