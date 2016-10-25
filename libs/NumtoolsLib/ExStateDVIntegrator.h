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
#ifndef _CODE_LIBS_NUMTOOLSLIB_NODE_EXSTATEDVINTEGRATOR_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_NODE_EXSTATEDVINTEGRATOR_INCLUDE_GUARD

#include <limits>
#include <vector>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>
#include "AbstractDVIntegratorCode.h"
#include "BasicDefinitions.h"
#include "DVIntegratorException.h"
#include "DVIntegratorStateParameter.h"

using std::vector;
using std::istream;
using std::ostream;
using std::string;


using std::numeric_limits;

namespace NumtoolsLib
{

	//! This is a wrapper class for the GSL numerical ODE solvers. Examples of its usage can be
	//! found in DynamicLib, for example in the Wilson-Cowan Algorithm. Unlike DVIntegrator it operates on a 
	//! state vector that is maintained by another class.
	//!
	//! The ExStateDVIntegrator is typically member of a class which requires the solution of a system
	//! of ODEs. In the source file of that class a function must be defined whose return type is int
	//! and whose arguments are given in the typedef for Function. A pointer to this function, (i.e the
	//! function name must be given in the constructor. If the GSL
	//! solver requires a Jacobian, then Derivative must also be defined, but for simple solvers,
	//! such as 4-th order Runge-Kutta this is not required and the derivative argument can be left out.
	//! Often, the ODE system requires parameters that need to be updated during the evolution of the system.
	//! This parameter can be a single number, a vector or anything else. The type of the parameter must
	//! be provided using the template argument. The class of the parameter must have a default constructor.
	template <class ParameterObject=double>
	class ExStateDVIntegrator : public AbstractDVIntegrator<ParameterObject>
	{
	public:

		//! Constructor
		ExStateDVIntegrator
		(
			Number,						//<! maximum number of integrations							
			double*,					//<! initial state
			Number,						//<! state size
			TimeStep,					//<! initial time step
			Time,						//<! initial time
			const Precision&,			//<! absolute and relative precision
			Function,					//<! dydt
			Derivative deravative = 0,  //<! Jacobian, if available
			const gsl_odeiv_step_type*	p_step_algorithm = gsl_odeiv_step_rkf45 //<! gsl integrator object
		);

		//! copy constructor must be defined, because the state of the integrator
		//! must be handled correctly
		ExStateDVIntegrator(const ExStateDVIntegrator&);

		//! Ditto for copy operator
		ExStateDVIntegrator& operator=(const ExStateDVIntegrator&);

		//! virtual destructor
		virtual ~ExStateDVIntegrator();

		//! Set a new state
		bool Reconfigure
		(
			const DVIntegratorStateParameter<ParameterObject>&
		);

		//! Current time
		Time CurrentTime() const;

		//! Only use if for efficiency reasons, acces to internal state is necessary
		const double* BeginState() const;

		//! Ditto
		const double* EndState  () const;

		//! streaming tag
		virtual string Tag () const;

		//! Direct access to the parameter object for performance reasons
		ParameterObject& Parameter();

	private:


	}; // end of NodeIntegrator

} // end of namespace

#endif // include guard
