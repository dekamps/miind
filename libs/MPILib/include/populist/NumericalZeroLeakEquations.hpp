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
#ifndef MPILIB_POPULIST_NUMERICALZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_NUMERICALZEROLEAKEQUATIONS_HPP_

#include <boost/shared_ptr.hpp>
#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/populist/AbstractZeroLeakEquations.hpp>
#include <MPILib/include/populist/AbstractRateComputation.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/LIFConvertor.hpp>
#include <MPILib/include/populist/NumericalZeroLeakParameter.hpp>
#include <MPILib/include/populist/ProbabilityQueue.hpp>

using boost::shared_ptr;
using NumtoolsLib::ExStateDVIntegrator;

namespace MPILib {
namespace populist {

//! Provides a numerical solution for the zero leak equations.
class NumericalZeroLeakEquations: public AbstractZeroLeakEquations {
public:


	NumericalZeroLeakEquations(Number&,	//!< reference to the current number of bins
			valarray<Potential>&,		//!< reference to state array
			Potential&,					//!< reference to the check sum variable
			SpecialBins&, PopulationParameter&,	//!< reference to the PopulationParameter
			PopulistSpecificParameter&,	//!< reference to the PopulistSpecificParameter
			Potential&				//!< reference to the current scale variable
			);

	virtual ~NumericalZeroLeakEquations() {
	}

	virtual void Configure(void*);

	//! Get input parameters at start of every Evolve
	virtual void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector);

	//! Adapt input parameters every simulation step
	virtual void AdaptParameters();

	virtual void RecalculateSolverParameters();

	virtual void Apply(Time);

	Rate CalculateRate() const;

	// overload to account for refractive probability
	virtual Probability RefractiveProbability() const {
		return _queue.TotalProbability();
	}

private:

	void InitializeIntegrators();
	void PushOnQueue(Time, double);
	void PopFromQueue(Time);

	Time _time_current;
	Number* _p_n_bins;
	PopulationParameter* _p_par_pop;
	valarray<Potential>* _p_array_state;
	Potential* _p_check_sum;
	LIFConvertor _convertor;
	auto_ptr<AbstractRateComputation> _p_rate_calc;

	boost::shared_ptr<ExStateDVIntegrator<NumericalZeroLeakParameter> > _p_integrator;
	boost::shared_ptr<ExStateDVIntegrator<NumericalZeroLeakParameter> > _p_reset;
	NumericalZeroLeakParameter _parameter;
	ProbabilityQueue _queue;
};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_NUMERICALZEROLEAKEQUATIONS_HPP_
