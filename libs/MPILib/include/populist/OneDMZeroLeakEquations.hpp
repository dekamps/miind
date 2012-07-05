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
#ifndef MPILIB_POPULIST_ONEDMZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_ONEDMZEROLEAKEQUATIONS_HPP_

#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>
#include <MPILib/include/populist/AbstractZeroLeakEquations.hpp>
#include <MPILib/include/populist/ABConvertor.hpp>
#include <MPILib/include/populist/OneDMInputSetParameter.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>

namespace MPILib {
namespace populist {

class AbConvertor;
class PopulistSpecificParameter;

class OneDMZeroLeakEquations: public AbstractZeroLeakEquations {
public:

	OneDMZeroLeakEquations(Number&,	//<! reference to the variable keeping track of the current number of bins
			std::valarray<Potential>&,		//<! reference to the state array
			Potential&, SpecialBins&, PopulationParameter&,	//!< serves now mainly to communicate t_s
			PopulistSpecificParameter&, Potential&//!< current potential interval covered by one bin, delta_v

			);

	virtual ~OneDMZeroLeakEquations();

	virtual void Configure(void*);

	virtual void Apply(Time);

	virtual void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector) {
		_convertor.SortConnectionvector(nodeVector, weightVector, typeVector);
	}
	virtual void AdaptParameters() {
		_convertor.AdaptParameters();
	}

	virtual void RecalculateSolverParameters() {
		_convertor.RecalculateSolverParameters();
	}

	virtual Time CalculateRate() const;

private:

	gsl_odeiv_system InitializeSystem() const;

	//! OneDMZeroLeakEquations does not need this, but the base class requires this
	InputParameterSet Set() {
		InputParameterSet set;
		return set;
	}

	Number& _n_bins;
	std::valarray<Potential>* _p_state;
	gsl_odeiv_system _system;		// moving frame prevents use of DVIntegrator

	ABConvertor _convertor;
	Number _n_max;

	gsl_odeiv_step* _p_step;
	gsl_odeiv_control* _p_control;
	gsl_odeiv_evolve* _p_evolve;

	gsl_odeiv_system _sys;

	OneDMInputSetParameter _params;

};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ONEDMZEROLEAKEQUATIONS_HPP_
