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

#ifndef _CODE_LIBS_GEOMLIB_ABSTRACTNEURALDYNAMICS_H_
#define _CODE_LIBS_GEOMLIB_ABSTRACTNEURALDYNAMICS_H_

#include <vector>
#include <MPILib/include/BasicDefinitions.hpp>
#include "CurrentCompensationParameter.hpp"
#include "OdeParameter.hpp"

namespace GeomLib {

	//! The configuration of a GeomAlgorithm requires that the neural dynamics is defined somewhere. The dynamics is used to define an OdeSystem.

	//!  There are predefined derived classes for leaky-integrate-and-fire and quadratic-integrate-and-fire
	//!  dynamics. The dynamics is usually defined on a grid, whose dimensions are specified in the OdeParameter. OdeParameter
	//!  also contains a NeuronParameter that determines the dynamics.Anyone who wants to use their own model of neural dynamics.
	//!  To define dynamics on this grid, the EvolvePotential must be overloaded. Derived classes exist for
	//!  LIF and QIF neurons, and more generally for spiking neurons (SpikingNeuralDynamics).

class AbstractNeuralDynamics {
	public:

		//! Constructor
		AbstractNeuralDynamics
		(
			const OdeParameter&,		//! Neuron parameter values, and other information relevant to binning structure
			const CurrentCompensationParameter& = CurrentCompensationParameter(0.,0.) //! By default, no current compensation, but if it's chosen, it directly affects the neural dynamics and hence the binning
		);

		//! Virtual destructor
		virtual ~AbstractNeuralDynamics() = 0;

		//! Given a potential, specify how it evolves over a given time step
		//! The range of validity of this function is determined by the overloaded function
		virtual Potential
			EvolvePotential
			(
				Potential,
				MPILib::Time
			) const = 0;

		//! virtual construction mechanism
		virtual AbstractNeuralDynamics* Clone() const  = 0;

		//! Provide efficient access. For use in time critical code
		const OdeParameter& Par() const {return _par; }

		//! Fundamental time step by which mass is shifted through the geometric bins. Consult
		//! the '1D document' for details.
		virtual MPILib::Time TStep() const = 0;

		//! Period for the dynamic model. Consult the '1D document'.
		virtual MPILib::Time TPeriod() const = 0;

		//! Generate the bin boundaries for geometric binning based on the dyn
                virtual std::vector<Potential> InterpretationArray() const = 0;

		//! Return the current compensation object; can be used to test whether current compensation is applied.
		const CurrentCompensationParameter& ParCur() const { return _par_current; }

	protected:

		//! Time critical access for derived classes
		const OdeParameter _par;

		//! For use in the concrete dynamics instantiation
		const CurrentCompensationParameter _par_current;
	};
}


#endif /* ABSTRACTNEURALDYNAMICS_H_ */
