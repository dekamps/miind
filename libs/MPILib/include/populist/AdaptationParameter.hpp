// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef MPILIB_POPULIST_ADAPTATIONPARAMETER_HPP_
#define MPILIB_POPULIST_ADAPTATIONPARAMETER_HPP_

#include <MPILib/include/populist/PopulistParameter.hpp>

namespace MPILib {
namespace populist {

	//! Parameter to store adaptation values for the 1DM Markov process of Muller et al. (2007)
	//! http://dx.doi.org/10.1162/neco.2007.19.11.2958

	//! At the moment the only variable relevant for this population are the adaptation time constant $t_s$,
	//! the adaptation jump value q and the maximum value of $g$ considered in the algorithm. 
	//! At the moment the base class variables are not used. The derivation is nonetheless required for two
	//! reasons: The effective
	//! values for the neuron are implicit in the a and b values that are used to drive the OneDMAlgorithm,
	//! but in the longer run it may be that these values will be interpolated from input parameters and neuron
	//! state variables, and then it would make sense to define the other parameters of the neuron population
	//! in here. A second reason for deriving from PopulationParameter is that it is easier to use the
	//! existing code of PopulationGridController.

	struct AdaptationParameter {

		//! default constructor
		AdaptationParameter():
		_t_adaptation(0),
		_q(0),
		_g_max(0)
		{}

		//! Constructor, adaptation parameters only
		AdaptationParameter
		(
			Time,
			State,
			State
		);

		Time	_t_adaptation;
		State	_q;
		State	_g_max;

		
	};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ADAPTATIONPARAMETER_HPP_
