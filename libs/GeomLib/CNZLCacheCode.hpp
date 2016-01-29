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

#include "CNZLCache.hpp"

using namespace GeomLib;
using namespace MPILib;

template <class Estimator>
CNZLCache<Estimator>::CNZLCache():
_old_set(0),
_vec_coverpair(0)
{
}

template <class Estimator>
void CNZLCache<Estimator>::InitializeCoverPairs()
{
	// create the vectors if necessary
	while (_vec_coverpair.size() < _p_vec_set->size()){		
		input_pair_list pair_list;
		pair_list.first  = vector<typename Estimator::CoverPair>(_n_bins);
		pair_list.second = vector<typename Estimator::CoverPair>(_n_bins);
		_vec_coverpair.push_back(pair_list);
	}

	// vectors are there
	for (Index i = 0; i < _p_vec_set->size(); i++){

		// initially this was only done when rate_e != 0; this is an error, as input steps may change at point in time when the rate happens to 0 incidently
		MPILib::Efficacy h_e = (*_p_vec_set)[i]._h_exc;
		for(Index j = 0; j < _n_bins; j++)
			_vec_coverpair[i].first[j] = _p_estimator->CalculateBinCover(j,-h_e);

		MPILib::Efficacy h_i = (*_p_vec_set)[i]._h_inh;
		for (Index j = 0; j < _n_bins; j++)
			_vec_coverpair[i].second[j] = _p_estimator->CalculateBinCover(j,-h_i);

	}
}

template <class Estimator>
bool CNZLCache<Estimator>::InputStepsHaveChanged() const
{
	const vector<InputParameterSet> vec_set = *_p_vec_set;

	if (_old_set.size() != _p_vec_set->size() )
		return true;
	
	Number n_size = _old_set.size();
	for (Index i = 0; i < n_size; i++)
	{
		if (vec_set[i]._h_exc != _old_set[i]._h_exc || vec_set[i]._h_inh != _old_set[i]._h_inh)
			return true;
	}

	return false;
}

template <class Estimator>
void CNZLCache<Estimator>::Initialize
(
	const AbstractOdeSystem&         sys,
	const vector<InputParameterSet>& vec_set,
	const Estimator&              estimator
)
{
	_n_bins = sys.NumberOfBins();

	_p_estimator = &estimator;
	_p_vec_set = &vec_set;
	_p_sys     = &sys;

	if (InputStepsHaveChanged() ){
		InitializeCoverPairs();
		_old_set = *_p_vec_set;
	}
	// else
		// keep the old values
}
