// Copyright (c) 2005 - 2014 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification,are permitted provided that the following conditions are
// met
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of
//      conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived
//      from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _CODE_LIBS_GEOMLIB_GEOMPARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_GEOMPARAMETER_INCLUDE_GUARD

#include <string>
#include <boost/shared_ptr.hpp>
#include "AbstractOdeSystem.hpp"
#include "CurrentCompensationParameter.hpp"
#include "DiffusionParameter.hpp"
#include "OdeParameter.hpp"

using std::string;

namespace GeomLib {

	struct GeomParameter {
	public:

		GeomParameter
		(
			const AbstractOdeSystem&,															//! Any OdeSystem object, this defines the neural model. This object will be cloned.
			Potential scale								= 1.0,  								//! In a diffusion interpretation of the input, it must be made clear what scale is considered to be small
			const DiffusionParameter& par_diff          = DiffusionParameter(0.03,0.03),		//! Typical values are 3% of the scale
			const CurrentCompensationParameter& par_cur = CurrentCompensationParameter(0.,0.),	//! By default, no current compensation
			const string& name_master					= "NumericalMasterEquation",			//! By default, a numerical scheme for the solution of the Master equation
			bool  no_master_equations               	= false									//! Do not run master equation when true
		);


		unique_ptr<AbstractOdeSystem>			_p_sys_ode;
		Potential								_scale;
		const DiffusionParameter&				_par_diff;
		const CurrentCompensationParameter		_par_cur;
		const string							_name_master;
		bool									_no_master_equation;
	};
}




#endif /* GEOMPARAMETER_H_ */
