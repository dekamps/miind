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
#ifndef MPILIB_POPULIST_ORNSTEINUHLENBECKPARAMETER_HPP_
#define MPILIB_POPULIST_ORNSTEINUHLENBECKPARAMETER_HPP_

#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {
namespace populist {

//! Parameters necessary for the configuration of an OUAlgorithm
//!
//! These are the parameters that define a leaky-integrate-and-fire neuron.

struct OrnsteinUhlenbeckParameter {

	Potential _theta = 0.0; //!< threshold potential in V
	Potential _V_reset = 0.0; //!< reset potential in V
	Potential _V_reversal = 0.0; //!< reversal potential in V
	Time _tau_refractive = 0.0; //!< (absolute) refractive time in s
	Time _tau = 0.0; //!< membrane time constant in s

	//! default constructor
	OrnsteinUhlenbeckParameter(){
	}

	//! standard constructor
	OrnsteinUhlenbeckParameter(Potential theta, Potential V_reset,
			Potential V_reversal, Time tau_refractive, Time tau) :
			_theta(theta), _V_reset(V_reset), _V_reversal(V_reversal), _tau_refractive(
					tau_refractive), _tau(tau) {
	}
};

typedef OrnsteinUhlenbeckParameter PopulationParameter;

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ORNSTEINUHLENBECKPARAMETER_HPP_
