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
#ifndef MPILIB_POPULIST_ZEROLEAKEQUATIONS_SINGLEINPUTZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_ZEROLEAKEQUATIONS_SINGLEINPUTZEROLEAKEQUATIONS_HPP_


#include <MPILib/include/populist/zeroLeakEquations/LIFZeroLeakEquations.hpp>

namespace MPILib {
namespace populist {
namespace zeroLeakEquations {

	class SingleInputZeroLeakEquations : public LIFZeroLeakEquations {

	public:
	/**
	 * default constuctor
	 * @param n_bins reference to the current number of bins
	 * @param array_state reference to state array
	 * @param check_sum reference to the check sum variable
	 * @param bins reference to bins variable: reversal bin, reset bin, etc
	 * @param par_pop reference to the PopulationParameter
	 * @param par_spec reference to the PopulistSpecificParameter
	 * @param delta_v reference to the current scale variable
	 * @param circ reference to the AbstractCirculantSolver
	 * @param noncirc reference to the AbstractNonCirculantSolver
	 */
		SingleInputZeroLeakEquations
		(
			Number&,							//!< reference to the current number of bins
			std::valarray<Potential>&,					//!< reference to state array
			Potential&,							//!< reference to the check sum variable
			SpecialBins&,			                                //!< object for the storage of specific bins, such as the reset bin 
			parameters::PopulationParameter&,	                  	//!< reference to the PopulationParameter 
			parameters::PopulistSpecificParameter&,	                	//!< reference to the PopulistSpecificParameter
			Potential&,	                                                //!< reference to the current scale variable
			const circulantSolvers::AbstractCirculantSolver&,               //!< reference to the circulant solver
			const nonCirculantSolvers::AbstractNonCirculantSolver&          //!< reference to the non circulant solver
		);

	/**
	 * Given input parameters, derived classes are free to implement their own solution for ZeroLeakEquations
	 * @param time The time
	 */
		virtual void Apply(Time);

	private:

		Time _time_current;
	};
}
}
}
#endif // include guard
