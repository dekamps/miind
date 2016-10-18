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
#ifndef _CODE_LIBS_GEOMLIB_QIFODESYSTEM_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_QIFODESYSTEM_INCLUDE_GUARD

#include "OdeParameter.hpp"
#include "QifParameter.hpp"
#include "SpikingOdeSystem.hpp"
#include "SpikingQifNeuralDynamics.hpp"

namespace GeomLib {

	//! A geometric grid based on Quadratic-Integrate-and Fire (QIF) dynamics

	//! This is a geometric grid based on the topology of spiking neuron dynamics. The
	// specific neural dynamics is QIF dynamics.

	class QifOdeSystem : public SpikingOdeSystem {
	public:

		//! standard constructor
		QifOdeSystem
		(
			const SpikingQifNeuralDynamics& //!< Accepts an object implementing the dynamics
		);

		//! copy constructor
		QifOdeSystem(const QifOdeSystem&);

		//! destructor
		virtual ~QifOdeSystem();

		//! evolve interpretation array over time
		virtual void 
			Evolve
			(
				Time				//! time to evolve
			);

		//! virtual copying mechanism
		virtual QifOdeSystem* Clone() const;

		//! return current firing rate
		virtual MPILib::Rate CurrentRate() const;

		//! return the DC contribution applied in current compensation
		virtual Potential DCContribution() const { return _par_qif.Gammasys() - _par_qif._gamma; }

	  virtual MPILib::Index CurrentIndex() const { return _index; }

		double TBatch() const { return _queue.TBatch(); }

	private:

		QifParameter	_par_qif;

	};
}

#endif

