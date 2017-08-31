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
#ifndef _CODE_LIBS_GEOMLIB_SPIKINGODESYSTEM_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_SPIKINGODESYSTEM_INCLUDE_GUARD

#include <MPILib/include/ProbabilityQueue.hpp>
#include "AbstractOdeSystem.hpp"
#include "SpikingNeuralDynamics.hpp"

using MPILib::populist::ProbabilityQueue;

namespace GeomLib {

	//! In this system of ordinary differential equations it is assumed that dynamics is always spiking.
	class SpikingOdeSystem : public AbstractOdeSystem {
	public:

		SpikingOdeSystem
		(
			const SpikingNeuralDynamics&
		);

		SpikingOdeSystem
		(
			const SpikingOdeSystem&
		);

		//! This is an abstract base class.
		virtual ~SpikingOdeSystem() = 0;

		void UpdateIndex() { if (--_index  == -static_cast<int>(_nr_bins) ) _index = 0; }

		//! virtual construction by the framework
		virtual SpikingOdeSystem* Clone() const = 0;

		virtual Potential DCContribution() const = 0;

	  virtual MPILib::Index CurrentIndex() const = 0;

	protected:

	  std::vector<MPILib::Index> InitializeCacheMap(MPILib::Number);
		void UpdateCacheMap();

	  MPILib::Index UpdateMapPotentialToProbabilityBin(MPILib::Index i) const;

	    MPILib::Number _nr_bins;
		int		       _index;

		ProbabilityQueue	_queue;

		void StoreInQueue      	();
		void RetrieveFromQueue	();

	};

  inline MPILib::Index SpikingOdeSystem::UpdateMapPotentialToProbabilityBin(MPILib::Index i) const {
		assert( i < this->Par()._nr_bins);
		int j = _index + i;
		return (j >= 0) ? i + _index : i + _index + _nr_bins;
	}
}

#endif // include guard

