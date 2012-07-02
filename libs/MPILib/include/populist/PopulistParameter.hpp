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
#ifndef MPILIB_POPULIST_POPULISTPARAMETER_HPP_
#define MPILIB_POPULIST_POPULISTPARAMETER_HPP_

#include <MPILib/include/populist/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/PopulistSpecificParameter.hpp>



namespace MPILib {
namespace populist {

	//! Auxiliary class that brings together the neuronal parameters and the specific 
	//! algorithmic parameters of the PopulistLib. 
	//!

	struct PopulistParameter {

		PopulationParameter			_par_pop;	//!< neuronal parameters
		PopulistSpecificParameter	_par_spec;	//!< grid, and algorithm parameters

		//! default constructor
		PopulistParameter(){}

		//! standard constructor
		PopulistParameter
		(
			const PopulationParameter& par_pop,
			const PopulistSpecificParameter& par_spec
		):
		_par_pop(par_pop),
		_par_spec(par_spec)
		{
		}
	};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_POPULISTPARAMETER_HPP_
