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
#ifndef _CODE_LIBS_POPULISTLIB_SPIKINGQIFDYNAMICS_H_
#define _CODE_LIBS_POPULISTLIB_SPIKINGQIFDYNAMICS_H_

#include "SpikingNeuralDynamics.hpp"
#include "QifParameter.hpp"

using MPILib::Time;

namespace GeomLib {

//! This class models the dynamics of  of Quadratic Integrate and Fire (QIF)
//! neurons

//! This model is discussed extensively in Izhikevich's book 'Dynamical Systems in Neuroscience'.
//! By a suitable choice of units a more complex equation can be simplified to a 'topological normal form':
//! \f[
//! \frac{dV}{dt} = I + V^2
//! \f]
//! Under this equation V may reach infinity in finite time. It is assumed that the neuron's potential will be reset
//! to \f$ V_{reset} \f$ after this time. In simulations it is necessary to chose a finite value \f$ V_{peak} \f$ as the
//! maximum that a neuron's membrane potential can attain. In the QIF model an instantaneous reset to \f$ V_{reset} \f$
//! will occur.  Configuration requires a QIFParameter and an OdeParameter.

	class SpikingQifNeuralDynamics : public SpikingNeuralDynamics{
	public:
		SpikingQifNeuralDynamics
		(
			const OdeParameter&,
			const QifParameter&		//! parameter specific  to QIF dynamics
		);

		SpikingQifNeuralDynamics
		(
			const SpikingQifNeuralDynamics&
		);

		virtual ~SpikingQifNeuralDynamics();

		//! calculate time to reach threshold for the given potential
		virtual Time TimeToInf(Potential) const;

		virtual Time TStep() const;

		virtual Time TPeriod() const;

		virtual Potential
			EvolvePotential
			(
				Potential,
				Time
			) const;

		virtual SpikingQifNeuralDynamics* Clone() const;

		const QifParameter& ParQif() const {return _par_qif; }

	protected:

		const QifParameter _par_qif;
	};
}



#endif /* SPIKINGQIFDYNAMICS_H_ */
