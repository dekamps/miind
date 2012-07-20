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
#ifndef MPILIB_POPULIST_LIFZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_LIFZEROLEAKEQUATIONS_HPP_

#include <MPILib/include/populist/AbstractZeroLeakEquations.hpp>
#include <MPILib/include/populist/AbstractRateComputation.hpp>
#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {
namespace populist {

//!! This is the base class for population density methods based on leaky-integrate-and-fire neurons

//!
class LIFZeroLeakEquations: public AbstractZeroLeakEquations {
public:


	LIFZeroLeakEquations( VALUE_REF_INIT
	Number&,						//!< reference to the current number of bins
			std::valarray<Potential>&,				//!< reference to state array
			Potential&,					//!< reference to the check sum variable
			SpecialBins&,//!< reference to bins variable: reversal bin, reset bin, etc
			parameters::PopulationParameter&,	//!< reference to the PopulationParameter
			parameters::PopulistSpecificParameter&,	//!< reference to the PopulistSpecificParameter
			Potential&,				//!< reference to the current scale variable
			const AbstractCirculantSolver&, const AbstractNonCirculantSolver&);

	virtual ~LIFZeroLeakEquations() {
	}

	virtual void Configure(void* p_void = 0);

	virtual void Apply(Time) {
	}

	virtual void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector);

	virtual void AdaptParameters();

	virtual void RecalculateSolverParameters();

	virtual Rate CalculateRate() const;

	//! Some  AbstractZeroLeakEquations have derived classes which keep track of refractive probability.
	//! These derived classes can overload this method, and make this amount available. For example,
	//! when rebinning this probability must be taken into account. See, e.g. RefractiveCirculantSolver.
	virtual Probability RefractiveProbability() const {
		return _p_solver_circulant->RefractiveProbability();
	}

protected:

	virtual void ScaleRefractiveProbability(double scale) {
		_p_solver_circulant->ScaleProbabilityQueue(scale);
	}

	void ApplyZeroLeakEquationsAlphaExcitatory(Time);

	void ApplyZeroLeakEquationsAlphaInhibitory(Time);

	Time _time_current;
	Number* _p_n_bins;
	std::valarray<Potential>* _p_array_state;
	Potential* _p_check_sum;
	LIFConvertor _convertor;
	boost::shared_ptr<AbstractCirculantSolver> _p_solver_circulant;
	boost::shared_ptr<AbstractNonCirculantSolver> _p_solver_non_circulant;
	boost::shared_ptr<AbstractRateComputation> _p_rate_calc;
};

} /* namespace populist */
} /* namespace MPILib */
#endif // include guard
