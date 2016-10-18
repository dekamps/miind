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
#ifndef _CODE_LIBS_NUMTOOLSLIB_NODE_GSLOBJECTS_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_NODE_GSLOBJECTS_INCLUDE_GUARD

#include <boost/shared_ptr.hpp>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>
#include "Precision.h"

using boost::shared_ptr;


namespace NumtoolsLib {
  typedef unsigned int Number;

	typedef int (*Function)   (double t, const double y[], double dydt[], void * params);
	typedef int (*Derivative) (double t, const double y[], double * dfdy, double dfdt[], void * params);	

	//! auxilliary struct to clean up some of the code of AbstractDVIntegrator
	struct GSLObjects{

		const gsl_odeiv_step_type*		_p_step_algorithm; //!< pointer to function, so no ownership issues
		gsl_odeiv_step*					_p_step;
		gsl_odeiv_control*				_p_controller;
		gsl_odeiv_evolve*				_p_evolver; 
		gsl_odeiv_system				_system;

		GSLObjects
		(
			const gsl_odeiv_step_type*,	//<! choice for solver type
			Number,						//<! state size
			const Precision&			//<! absolute and relative precision conform GSL documentation
		);

		~GSLObjects();

	};
}

#endif // include guard

