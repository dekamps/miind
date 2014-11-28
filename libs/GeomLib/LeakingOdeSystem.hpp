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
#ifndef _CODE_LIBS_GEOMLIB_LEAKINGODESYSTEM_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_LEAKINGODESYSTEM_INCLUDE_GUARD

#include <GeomLib/AbstractOdeSystem.hpp>
#include <vector>
#include "LifNeuralDynamics.hpp"
using std::vector;

namespace GeomLib {

  //! System for neural dynamics of a strictly leaky nature.

  //! This System is used for neural dynamics that is leaky, such as leaky-integrate-and-fire (LIF) neurons. Use
  //! this controller if all you ever want is model populations of LIF neurons. It is then to be preferred over
  //! SpikingOdeSystem as it does not rely on the current compensation trick and does not introduce a diffusive
  //! background. The number of bins that is specified in the OdeParameter refers to the interval [V_r, V_th).
  //! If the user specifies a V_min < V_th, the system will compute the number of extra bins to required to accommodate this
  //! part of state space, and bin arrays will be larger than the number of bins specified by the user. 
  class LeakingOdeSystem  : public AbstractOdeSystem{
  public:

	//! Use a LifNeuralDynamics object to create the geometric binning
    LeakingOdeSystem
     (
    		const LifNeuralDynamics&
     );

    //! Copy constructor
    LeakingOdeSystem
    (
    		const LeakingOdeSystem&
    );

    //! Efficient access to the OdeParameter values
    const OdeParameter& Par() const { return _par; }

    //! virtual constructor mechanism
    virtual LeakingOdeSystem* Clone() const;

    //! Maps a given mass bin to its current potential interval; 0 <= index < Par()._n_bins
    Index MapProbabilityToPotentialBin(Index) const;

    //! Move the mass to a new bin (one step of decay)
	void UpdateIndex() { _index++;}

	//!
	virtual void
		Evolve
		(
			MPILib::Time
		);


	virtual MPILib::Rate CurrentRate() const;


	pair<Number,Number> BinDistribution() const;


  private:



    void   ReversalBinScoop		();
    void   UpdateCacheMap		();
    void   UpdateCacheMapReverse();
    Index  UpdateMapProbabilityToPotentialBin(Index i);
    Index  UpdateMapPotentialToProbabilityBin(Index i);

    const LifNeuralDynamics	_dyn;

    Number _N_pos;
    Number _N_neg;
    Number _N_bins;

	int		_index;

	vector<Index> _map_cache_reverse;

  };

  inline Index LeakingOdeSystem::MapProbabilityToPotentialBin(Index i) const {
	  assert (i < _map_cache.size());
	  return _map_cache_reverse[i];
  }

} // include guard

#endif
