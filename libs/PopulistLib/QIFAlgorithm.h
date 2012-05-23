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

#ifndef _CODE_LIBS_POPULISTLIB_QIFALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_QIFALGORITHM_INCLUDE_GUARD

#include "QIFParameter.h"
#include "CharacteristicAlgorithm.h"

namespace PopulistLib {

	//! This algorithm models the probability  distribution function of a population of Quadratic Integrate and Fire (QIF)
	//! neurons

	//! This model is discussed extensively in Izhikevich's book 'Dynamical Systems in Neuroscience'. It can be 'derived' as an 
	//! approximation of a conductance-based model dominated by a persistent sodium current,  \f$ I_{Na,p}\f$. This current is responsible
	//! for the upstroke of the neuron membrane potential during a spike. As a neuronal model it is characterised by two different
	//! regimes: one where it represents Class I neurons, and another one where it represents Class II neurons. Class I neurons show
	//! a graded respons to input current, reflected in a firing rate that is commensurate to the input stimulation, whereas Class II
	//! neurons respond to input with action potentials in a certain frequency band that is 'relatively insensitive to the strength
	//! of the applied current.'

	//! Assuming \f$ I_{Na,p} \f$ to be instantaneous, one can write the equation for the persistent sodium model
	//! \f[
	//! C \frac{d V}{d t} = I - g_L(V - E_L) - g_{Na}m_{\infty}(V)(V - E_{Na} )
	//! \f]
	//! with
	//! \f[
	//! m_{\infty} = \frac{1}{1 + \exp\{ (V_{1/2} - V)/k \}} )
	//! \f]
	//! The resulting steady state current has a maximum approximately at \f$ (V_{sn}, I_{sn})  = (-46.6 mV, 16 pA) \f$ and can be approximated
	//! locally by:
	//! \f[
	//! I = I_{sn} - \gamma (V - V_{sn})^2
	//! \f]
	//! with \f$ \gamma \approx 0.45 \mbox{nS/mV} \f$.
	//! By a suitable change of units the resulting equation can easily be cast into its 'topological normal form':
	//! \f[
	//! \frac{dV}{dt} = I + V^2
	//! \f]
	//! Under this equation V may reach infinity in finite time. It is assumed that the neuron's potential will be reset
	//! to \f$ V_{reset} \f$ after this time. In simulations it is necessary to chose a finite value \f$ V_{peak} \f$ as the
	//! maximum that a neuron's membrane potential can attain. In the QIF model an instantaneous reset to $\f V_{reset} \f$
	//! will occur.

	template <class Weight>
	class QIFAlgorithm : public CharacteristicAlgorithm<Weight> {
	public:

		//! Standard way for user to create algorithm
		QIFAlgorithm
		(
			const QIFParameter&,				//! Quadratic-integrate-and-fire specific parameter
			const CharacteristicParameter&		//! Generic parameter, required for all CharacteristicAlgorithm types, for example, to set the bin size
		);

		//! Copy constructor
		QIFAlgorithm(const QIFAlgorithm&);

		//! virtual destructor
		virtual ~QIFAlgorithm();

		//! An algorithm saves its log messages, and must be able to produce them for a Report
		virtual string LogString() const;

		//! Cloning operation, to provide each DynamicNode with its own 
		//! Algorithm instance
		virtual QIFAlgorithm<Weight>* Clone() const;

		//! A complete serialization of the state of an Algorithm, so that
		//! it can be resumed from its disk representation. NOT IMPLEMENTED.
		virtual bool Dump(ostream&) const ;

	};
}

#endif // include guard