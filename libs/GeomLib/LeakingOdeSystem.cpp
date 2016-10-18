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
#include <cassert>
#include "LeakingOdeSystem.hpp"
#include "GeomLibException.hpp"
#include "../NumtoolsLib/NumtoolsLib.h"

using namespace NumtoolsLib;
using namespace GeomLib;

LeakingOdeSystem::LeakingOdeSystem
(
 const LifNeuralDynamics& dyn

 ):
 AbstractOdeSystem		(dyn),
 _dyn					(dyn),
 _N_pos					(dyn.Npos()),
 _N_neg					(dyn.Nneg()),
 _N_bins				(_N_pos+_N_neg),
  _index				(0)
{
  _map_cache         = vector<MPILib::Index>(_N_bins);
	_map_cache_reverse = vector<MPILib::Index>(_N_bins);

	for (MPILib::Index i = 0; i < _N_bins; i++)
		_map_cache[i] = i;
	for (MPILib::Index i = 0; i < _N_bins; i++)
		_map_cache_reverse[i] = i;

}


LeakingOdeSystem::LeakingOdeSystem(const LeakingOdeSystem& sys):
AbstractOdeSystem		(sys._dyn),
_dyn					(sys._dyn),
_N_pos					(sys._N_pos),
_N_neg					(sys._N_neg),
_N_bins					(sys._N_bins),
_index					(0), // do not copy the state but force the start of a new simulation
_map_cache_reverse		(sys._map_cache_reverse)
{
  _map_cache         = std::vector<MPILib::Index>(_N_bins);
  _map_cache_reverse = std::vector<MPILib::Index>(_N_bins);

	for (MPILib::Index i = 0; i < _N_bins; i++)
		_map_cache[i] = i;
	for (MPILib::Index i = 0; i < _N_bins; i++)
		_map_cache_reverse[i] = i;

}

LeakingOdeSystem* LeakingOdeSystem::Clone() const
{
	return new LeakingOdeSystem(*this);
}

std::pair<MPILib::Number, MPILib::Number> LeakingOdeSystem::BinDistribution() const
{
  std::pair<MPILib::Number,MPILib::Number> pair_ret;
	pair_ret.first  = _N_pos;
	pair_ret.second = _N_neg;
	return pair_ret;
}

void LeakingOdeSystem::Evolve
(
	MPILib::Time t
)
{
  if ( ! NumtoolsLib::IsApproximatelyEqualTo(t, _t_step, 1e-9 ) )
		throw GeomLibException("QIFOdeSystem is designed to only have one fixed time step");

	this->UpdateIndex			();
	this->UpdateCacheMap		();
	this->UpdateCacheMapReverse	();
	this->ReversalBinScoop		();
	_t_current += t;
}

MPILib::Rate LeakingOdeSystem::CurrentRate() const
{
	return 0.0;
}

void LeakingOdeSystem::ReversalBinScoop()
{
  MPILib::Index i_rev = this->MapPotentialToProbabilityBin(_i_reversal);
  MPILib::Index i_th  = this->MapPotentialToProbabilityBin(_N_pos +_N_neg - 1);

	_buffer_mass[i_rev] += _buffer_mass[i_th];
	_buffer_mass[i_th] = 0.0;

	if (_N_neg > 0){
		i_rev        = this->MapPotentialToProbabilityBin(_i_reversal-1);
		MPILib::Index i_vmin = this->MapPotentialToProbabilityBin(0);
		_buffer_mass[i_rev] += _buffer_mass[i_vmin];
		_buffer_mass[i_vmin] = 0.0;
	}
}

void LeakingOdeSystem::UpdateCacheMapReverse()
{
  for (MPILib::Index i = 0; i < _N_bins; i++){
		_map_cache_reverse[i] = UpdateMapProbabilityToPotentialBin(i);
		assert(_map_cache[_map_cache_reverse[i]] == i);
	}
}

void LeakingOdeSystem::UpdateCacheMap(){
  for (MPILib::Index i = 0; i < _N_bins; i++)
		_map_cache[i] = UpdateMapPotentialToProbabilityBin(i);
}


MPILib::Index LeakingOdeSystem::UpdateMapPotentialToProbabilityBin(MPILib::Index i)
{
	assert( i < _dyn.Npos() + _dyn.Nneg());
	return (i >= _i_reversal )?
			modulo( (i - _N_neg + _index), _N_pos) + _N_neg :
			modulo( i - _index, _N_neg );
}

MPILib::Index LeakingOdeSystem::UpdateMapProbabilityToPotentialBin(MPILib::Index i)
{
	  int r = static_cast<int>(i) - static_cast<int>(_N_neg);
	  return ( r >= 0) ?
			  _N_neg + modulo(( r - _index ),(_N_pos) ) :
			  _N_neg - static_cast<int>(modulo(-r - 1 - _index,_N_neg)+1);
}
