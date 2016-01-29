// Copyright (c) 2005 - 2014 Marc de Kamps
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
#ifndef _CODE_LIBS_GEOMLIB_MASTEREQUATIONS_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_MASTEREQUATIONS_INCLUDE_GUARD

#include <MPILib/include/populist/ProbabilityQueue.hpp>
#include "../NumtoolsLib/NumtoolsLib.h"
#include "AbstractMasterEquation.hpp"
#include "AbstractOdeSystem.hpp"
#include "BinEstimator.hpp"
#include "CNZLCache.hpp"
#include "DiffusionParameter.hpp"
#include "CurrentCompensationParameter.hpp"
#include "GeomInputConvertor.hpp"
#include "MasterParameter.hpp"
#include "OdeParameter.hpp"

using MPILib::Rate;
using MPILib::Time;
using MPILib::Density;
using NumtoolsLib::ExStateDVIntegrator;

namespace GeomLib {

	//! Solves the Poisson Master equation on the probability mass bins.

	//! The Poisson master equation is determined by the combination
	//! of synaptic efficacy h and potential grid. Transitions occur
	//! away from this bin and probability enters from density a step
	//! h away. We calculate a transition matrix by translating the
	//! current bin and finding out which bins it covers. The system
	//! of equations that is described in http://arxiv.org/abs/1309.1654
	//! is then set up and solved.

	template <class Estimator = BinEstimator>
	class NumericalMasterEquation : public AbstractMasterEquation {
	public:

		using AbstractMasterEquation::sortConnectionVector;

		NumericalMasterEquation
		(
			AbstractOdeSystem&,   			 //!< OdeSystem defining the geometric grid
			const DiffusionParameter&,	       	 //!< Parameters that determine whether to use single or double input in Gaussian white noise approximation.
			const CurrentCompensationParameter&      //!< If current compensation is to be used, these are the values.
		);

		//! Destructor
		virtual ~NumericalMasterEquation();

		//! Apply the master equations over a period of time
		virtual void apply(MPILib::Time);


		//! Interpretation of the input is done by a GeomInputConvertor object.
		//! This is a hook to provide that object with the required information
		virtual void sortConnectionVector
		(
			const std::vector<MPILib::Rate>&,
			const std::vector<MPILib::DelayedConnection>&,
			const std::vector<MPILib::NodeType>&
		);

		virtual Rate getTransitionRate() const;

		Rate IntegralRate() const;

	private:

		NumericalMasterEquation(const NumericalMasterEquation&);
		NumericalMasterEquation& operator=(const NumericalMasterEquation&);

		void InitializeIntegrator();
		Density RecaptureProbability();

		Density Checksum() const;
		AbstractOdeSystem&	      	       	_system;
		const DiffusionParameter	       	_par_diffusion;
		const CurrentCompensationParameter	_par_current;

		Time			       	       	_time_current;

		Estimator		               	_estimator;
		GeomInputConvertor	       		_convertor;
	    CNZLCache<Estimator>    		_cache;

	    ExStateDVIntegrator<MasterParameter<Estimator> >    _integrator;
 
		ProbabilityQueue      			_queue;
		Rate			      	       	_rate;

		static Number	           		_max_iteration;
	    mutable vector<Density>	        _scratch_dense;
		mutable vector<Potential>	   	_scratch_pot;
	};
}

#endif // include guard

