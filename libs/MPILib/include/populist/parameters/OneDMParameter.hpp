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
#ifndef MPILIB_POPULIST_ONEDMPARAMETER_HPP_
#define MPILIB_POPULIST_ONEDMPARAMETER_HPP_

#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>
#include <MPILib/include/populist/parameters/AdaptationParameter.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>

namespace MPILib {
namespace populist {

	struct OneDMParameter {

		/**
		 * default constructor
		 */
		OneDMParameter()=default;

		/**
		 * constructor
		 * @param par_pop Defensive, serves no purpose atm
		 * @param par_adapt Adaptation specific parameter
		 * @param par_spec Grid and algorithm related stuff
		 */
		OneDMParameter
		(
				const PopulationParameter&			par_pop,
				const AdaptationParameter&			par_adapt,
				const PopulistSpecificParameter&	par_spec
		);

		/**
		 * Defensive, serves no purpose atm, but probably will do in the future
		 */
		PopulationParameter			_par_pop;
		/**
		 * Adaptation specific parameter; accidently also an AdaptationParameter
		 */
		AdaptationParameter			_par_adapt;
		/**
		 * Grid and algorithm related stuff
		 */
		PopulistSpecificParameter	_par_spec;

	};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ONEDMPARAMETER_HPP_
