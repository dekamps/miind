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
#include "SpikingOdeSystem.hpp"
#include "BasicDefinitions.hpp"

using namespace GeomLib;

SpikingOdeSystem::SpikingOdeSystem
(
	const SpikingNeuralDynamics& dyn
):
AbstractOdeSystem
(
 dyn
),
_index              (0),
_queue	       	    (TIME_QUEUE_BATCH)
{
  Number nr_bins = this->NumberOfBins();
  _map_cache = this->InitializeCacheMap(nr_bins);
}

SpikingOdeSystem::SpikingOdeSystem
(
	const SpikingOdeSystem& rhs
):
AbstractOdeSystem       (rhs),
_index                  (rhs._index),
_queue		        	(rhs._queue)
{
   Number nr_bins = this->NumberOfBins();
   _map_cache = this->InitializeCacheMap(nr_bins);
}

SpikingOdeSystem::~SpikingOdeSystem()
{
}

void SpikingOdeSystem::StoreInQueue()
{
	Number n_bins = this->NumberOfBins();
	int i_th = (_index+n_bins)%n_bins;
	MPILib::Probability p = _buffer_mass[i_th];
	MPILib::populist::StampedProbability prob;
	prob._prob = p;
	prob._time = _t_current + this->Par()._par_pop._tau_refractive;
	_queue.push(prob);
	_buffer_mass[i_th] = 0.0;
}

void SpikingOdeSystem::RetrieveFromQueue()
{
	MPILib::Probability p = _queue.CollectAndRemove(_t_current + _queue.TBatch());
	Number n_bins = this->NumberOfBins();
	int i_reset = (_i_reset + _index + n_bins)%n_bins;
	_buffer_mass[i_reset] += p;
}

vector<Index> SpikingOdeSystem::InitializeCacheMap(Number n){
  vector<Index> vec_ret(n);

  for (Index i = 0; i < n; i++)
    vec_ret[i] = i;

  return vec_ret;
}

void SpikingOdeSystem::UpdateCacheMap(){

  Number n_bins = this->NumberOfBins();

  for (Index i = 0; i < n_bins; i++)
    _map_cache[i] = this->UpdateMapPotentialToProbabilityBin(i);
}
