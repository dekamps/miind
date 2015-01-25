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
#ifndef _CODE_LIBS_GEOMLIB_CURRENTCOMPENSATION_PARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_CURRENTCOMPENSATION_PARAMETER_INCLUDE_GUARD

#include<cassert>
#include <MPILib/include/BasicDefinitions.hpp>

namespace GeomLib {

	//! Parameter for setting current compensation values for the neural models that use it
	struct CurrentCompensationParameter {

		MPILib::Potential _I;			//!< DC contribution
		MPILib::Potential _sigma;		//!< variability of the Poisson emulation


		//! Standard constructor
		CurrentCompensationParameter
		(
			MPILib::Potential I = 0,    //!< current
			MPILib::Potential sigma = 0 //!< variability
		):
		_I(I),
		_sigma(sigma)
		{
			assert( ! (I != 0 && sigma == 0.0 ));
		}

		//! Many models will not use CurrentCompensation; the convention is that as long as sigma is positive
		//! compensation will be applied, so if you do not want current compensation is applied keep both I and sigma
		//! equal to 0
		bool NoCurrentCompensation() const { return (_I == 0.0 && _sigma == 0.0) ? true : false; }

	};
}

#endif

