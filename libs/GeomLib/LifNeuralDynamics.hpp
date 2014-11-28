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
#include "AbstractNeuralDynamics.hpp"

#ifndef _CODE_LIBS_GEOMLIB_LIFNEURALDYNAMICS
#define _CODE_LIBS_GEOMLIB_LIFNEURALDYNAMICS

namespace GeomLib {

	//! Leaky-integrate-and-fire dynamics for LeakingOdeSystem

	//! This class provides leaky-integrate-and-fire dynamics for a LeakingOdeSystem. The LeakingOdeSystem
	//! provides the geometric binning for a leaky-integrate-and-fire population as modelled by GeomAlgorithm.
	//! It is the default choice for modelling populations of LIF neurons.


	class LifNeuralDynamics : public AbstractNeuralDynamics {
	public:

		//! The lambda parameter is creating the artificial period necessary for implementing a 'period'
		//! in leaky dynamics. Consult the '1D document.' This type of dynamics does not allow current compensation.
		LifNeuralDynamics
		(
			const OdeParameter&,	//! parameter specifying neural dynamics, bin size, etc.
			double lambda			//! recommended to be of the order 0.01; consult '1D document' for information on trade off between speed and efficiency
		);

		//! Destructor; required to be virtual
		virtual ~LifNeuralDynamics();

		//! Evolution according to leaky-integrate-and-fire dynamics
		virtual Potential
			EvolvePotential
			(
				MPILib::Potential,
				MPILib::Time
			) const;

		//! Fundamental time step by which mass is shifted through the geometric bins
		virtual MPILib::Time TStep() const;

		//! Time it takes for probability mass to go full cycle. For LIF dynamics this is close to the
		//! time it takes from threshold to lambda.
		virtual MPILib::Time TPeriod() const { return _t_period; }

		//! Virtual constructor mechanism
		virtual LifNeuralDynamics* Clone() const;

		//! Produce an array that contains the bin limits. The are contains the lower bin limits
		//! the highest bin limit of the highest bin is the threshold value of the neuron, theta
		//! which is not included in the array.
		virtual vector<Potential> InterpretationArray() const;

		//! Number of bins in the grid
		Number NumberOfBins() const {return _N_pos + _N_neg;}

		//! Number of bins above the reversal bin
		Number Npos() const { return _N_pos; }

		//! Number of bins below the reversal bin
		Number Nneg() const { return _N_neg; }

	private:

		Number Nposinit() const;
		Number Nneginit() const;
		MPILib::Time TimePeriod() const;

		double _lambda;

		MPILib::Time 	_t_period;
		Number 	_N_pos;
		MPILib::Time 	_t_step;
		Number 	_N_neg;
	};

}




#endif /* LIFNEURALDYNAMICS_H_ */
