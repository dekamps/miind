// Copyright (c) 2005 - 2011 Marc de Kamps
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
#ifndef _GEOM_LIBS_GEOMLIB_NEURONPARAMETER_INCLUDE_GUARD
#define _GEOM_LIBS_GEOMLIB_NEURONPARAMETER_INCLUDE_GUARD


#include <MPILib/include/BasicDefinitions.hpp>
#include "GeomLibException.hpp"

namespace GeomLib {


	//! Parameters necessary for the configuration of an OUAlgorithm
	//!
	//! These are the parameters that define a leaky-integrate-and-fire neuron.

	struct NeuronParameter  {

		MPILib::Potential _theta;				//!< threshold potential in V
		MPILib::Potential _V_reset;				//!< reset potential in V
		MPILib::Potential _V_reversal;			//!< reversal potential in V
		MPILib::Time      _tau_refractive;		//!< (absolute) refractive time in s
		MPILib::Time      _tau;					//!< membrane time constant in s

		//! default constructor
		NeuronParameter():
		_theta(0),
		_V_reset(0),
		_V_reversal(0),
		_tau_refractive(0),
		_tau(0){
			}

		//! standard constructor
		NeuronParameter
			(
				MPILib::Potential theta,
				MPILib::Potential V_reset,
				MPILib::Potential V_reversal,
				MPILib::Time      tau_refractive,
				MPILib::Time      tau
			):
		_theta(theta),
		_V_reset(V_reset),
		_V_reversal(V_reversal),
		_tau_refractive(tau_refractive),
		_tau(tau){
			if (_V_reset > theta || _V_reversal > theta)
				throw GeomLibException("Threshold should be largest potential");
			if (_tau_refractive > _tau)
				throw GeomLibException("tau_ref > tau");
		}

	};


} // end of GeomLib

#endif // include guard
