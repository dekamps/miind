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
#ifndef MPILIB_POPULIST_ABSTRACTRATECOMPUTATION_HPP_
#define MPILIB_POPULIST_ABSTRACTRATECOMPUTATION_HPP_

#include <valarray>

#include <MPILib/include/populist/InputParameterSet.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/BasicTypes.hpp>

using std::valarray;

namespace MPILib {
namespace populist {

	//! AbstractRateComputation
	//! There are several methods to calculate a Population's firing rate from the population density
	//! A virtual base class is provide, so that the methods can be exchanged in run time and the different
	//! methods can be compared within a single simulation

	class AbstractRateComputation {
	public:

		//! constructor
		AbstractRateComputation();

		virtual void Configure
		(
			// not const since a pointer to an element is needed, it should have been const otherwise:
			valarray<Density>&,			// state array
			const InputParameterSet&,	// current input to population
			const PopulationParameter&,	// neuron parameters
			Index						// index reversal bin
		);

		virtual ~AbstractRateComputation() = 0;

		virtual AbstractRateComputation* Clone() const = 0;

		virtual Rate CalculateRate
		(
			Number    // number current bins
		) = 0;

	protected:

		bool DefineRateArea
		(
			Potential
		);

		Potential BinToCurrentPotential(Index);

		Index						_index_reversal;
		valarray<Density>*			_p_array_state;
		const InputParameterSet*	_p_input_set;
		PopulationParameter			_par_population;
		valarray<Potential>			_array_interpretation;         

		Number		_n_bins;
		Number		_number_integration_area;      
		Index		_start_integration_area;       //

		Potential	_delta_v_rel;
		Potential	_delta_v_abs;


	private:


	};

} /* namespace populist */
} /* namespace MPILib */
#endif // end of include guard MPILIB_POPULIST_ABSTRACTRATECOMPUTATION_HPP_
