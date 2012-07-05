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
#ifndef MPILIB_POPULIST_OLDLIFZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_OLDLIFZEROLEAKEQUATIONS_HPP_

#include <MPILib/include/populist/LIFZeroLeakEquations.hpp>
#include <MPILib/include/BasicTypes.hpp>


namespace MPILib {
namespace populist {

	//! \deprecated DEPRECATED! In response to the discivery in (deKamps, 2006) that probability density sometimes must be transported from one bin to 
	//! a point between two bins a quick hack was devised, essentially running the NonCirculantSolver twice, using the time to express the proportionality
	//! of each bin. This is ugly and doubles simulation time.  OldZeroLeakEquations will not be available for use in the XML version of MIIND.
	class OldLIFZeroLeakEquations : public LIFZeroLeakEquations {
	public:

		typedef AbstractAlgorithm<PopulationConnection>::predecessor_iterator predecessor_iterator;

		OldLIFZeroLeakEquations
		(
			VALUE_REF_INIT
			Number&,								//!< reference to the current number of bins
			valarray<Potential>&,					//!< reference to state array
			Potential&,								//!< reference to the check sum variable
			SpecialBins&,							//!< reference to bins variable: reversal bin, reset bin, etc		
			PopulationParameter&,					//!< reference to the PopulationParameter 
			PopulistSpecificParameter&,				//!< reference to the PopulistSpecificParameter
			Potential&,								//!< reference to the current scale variable
			const AbstractCirculantSolver&,
			const AbstractNonCirculantSolver& 
		);

		virtual ~OldLIFZeroLeakEquations(){}

		//! No-op for OldLIFZeroLeakEquations
		virtual void Configure
		(
			void* p_void = 0
		);

		virtual void Apply(Time);

		virtual void SortConnectionvector
		(
			predecessor_iterator,
			predecessor_iterator
		);

		virtual void AdaptParameters
		(
		);

		virtual void RecalculateSolverParameters();

		virtual Rate CalculateRate() const;

	private:

		void ApplyZeroLeakEquationsAlphaExcitatory
		(
			Time
		);

		void ApplyZeroLeakEquationsAlphaInhibitory
		(
			Time
		);

		Time									_time_current;
		Number*									_p_n_bins;
		valarray<Potential>*					_p_array_state;
		Potential*								_p_check_sum;
		LIFConvertor							_convertor;
		auto_ptr<AbstractCirculantSolver>		_p_solver_circulant;
		auto_ptr<AbstractNonCirculantSolver>	_p_solver_non_circulant;
		auto_ptr<AbstractRateComputation>		_p_rate_calc;
	};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_OLDLIFZEROLEAKEQUATIONS_HPP_
