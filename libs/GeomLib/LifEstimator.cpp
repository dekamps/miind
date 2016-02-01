// Copyright (c) 2005 - 2016 Marc de Kamps
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

#include "LifEstimator.hpp"
using namespace GeomLib;
using namespace std;


LifEstimator::LifEstimator
(
	const vector<double>& vec_interpretation,
	const OdeParameter& par_ode
):
_par_ode(par_ode),
_vec_interpretation(vec_interpretation),
_i_reversal(IndexReversalBin()),
_lambda(GetLambda()),
_t_period(this->TPeriod()),
_N_pos(Nposinit()),
_t_step(this->TStep()),
_N_neg(Nneginit())
{
}

Number LifEstimator::Nneginit() const
{
  float n_extra = (_par_ode._par_pop._tau/_t_step)*log((_par_ode._par_pop._theta - _par_ode._par_pop._V_reversal)/(_par_ode._par_pop._V_reversal - _par_ode._V_min));
  Number N_min = static_cast<Number>(ceil(_N_pos - (n_extra)));
  return N_min;
}

Index LifEstimator::IndexReversalBin() const
{
	// find the reversal potential. It must be in the interpretation array
	auto it = find(_vec_interpretation.begin(),_vec_interpretation.end(),_par_ode._par_pop._V_reversal);
	if (it == _vec_interpretation.end())
		throw GeomLibException("Reversal potential not in interpretation array.");
	return it - _vec_interpretation.begin();
}
double LifEstimator::GetLambda() const
{

	Potential V_plus = _vec_interpretation[_i_reversal+1];
	return (V_plus - _par_ode._par_pop._V_reversal )/(_par_ode._par_pop._theta - _par_ode._par_pop._V_reversal);
}

Number LifEstimator::Nposinit() const
{
  return _par_ode._nr_bins;
}


MPILib::Time LifEstimator::TStep() const
{
  return _t_period/(_N_pos - 1);
}

MPILib::Time LifEstimator::TPeriod() const
{
  assert( _par_ode._V_min   <= _par_ode._par_pop._V_reset);
  assert( _par_ode._par_pop._V_reset <  _par_ode._par_pop._theta );

  double V_plus = _par_ode._par_pop._V_reversal + _lambda*(_par_ode._par_pop._theta - _par_ode._par_pop._V_reversal);
  return _par_ode._par_pop._tau*log((_par_ode._par_pop._theta -_par_ode._par_pop._V_reversal)/(V_plus - _par_ode._par_pop._V_reversal));
}

int LifEstimator::SearchBin(Potential V) const
{
	if (V <_par_ode._V_min || V > _par_ode._par_pop._theta)
		return -1;

	if (V >= _vec_interpretation.back())
		return _N_pos + _N_neg - 1;

	Potential V_plus  = _vec_interpretation[_i_reversal+1];
	Potential V_rev   = _vec_interpretation[_i_reversal];
	Potential V_th    = _par_ode._par_pop._theta;
	MPILib::Time tau  = _par_ode._par_pop._tau;

	if (V >= V_rev  && V < V_plus)
		return _i_reversal;

	if (V >= V_rev - V_plus && V < V_rev)
		return _i_reversal - 1;

	if ( V <  V_rev - V_plus){
		int n = static_cast<int>(floor(tau*log((V_rev - V)/V_plus)/_t_step));
		// _Neg - 1 - (n+1)
		return _N_neg - n - 2;
	}

	if ( V >= V_plus){
		int n = static_cast<int>(floor(tau*log((V_th - V_rev)/(V - V_rev))/_t_step));
		return _N_pos + _N_neg - 1 - n;
	}
	throw GeomLibException("SearchBin leaks.");
}

double LifEstimator::BinLowFraction(Potential v, int i_tr_low) const {
	int n = _vec_interpretation.size();
	if (i_tr_low < 0 || i_tr_low > n-1)
		return 0.0;

	Potential low_bin  = _vec_interpretation[i_tr_low];
	Potential high_bin = (i_tr_low != static_cast<int>(_vec_interpretation.size()) - 1)? _vec_interpretation[i_tr_low + 1] : _par_ode._par_pop._theta;
	assert( low_bin <= v && high_bin >= v);
	return  (high_bin -v )/(high_bin - low_bin);
}

double LifEstimator::BinHighFraction(Potential v, int i_tr_high) const {
	int n = _vec_interpretation.size();
	if (i_tr_high < 0 || i_tr_high > n-1)
		return 0.0;

	Potential low_bin  = _vec_interpretation[i_tr_high];
	Potential high_bin = (i_tr_high != static_cast<int>(_vec_interpretation.size()) - 1)? _vec_interpretation[i_tr_high + 1] : _par_ode._par_pop._theta;
	assert( low_bin <= v && high_bin >= v);
	return (v - low_bin)/(high_bin - low_bin);
}

LifEstimator::CoverPair LifEstimator::CalculateBinCover(Index i, Potential delta_v) const{
	assert(i < _vec_interpretation.size());
	CoverPair pair_ret;
	Potential low = _vec_interpretation[i];
	Potential high = (i != _vec_interpretation.size() - 1) ? _vec_interpretation[i+1] : _par_ode._par_pop._theta;
	Potential trans_low  = low  + delta_v;
	Potential trans_high = high + delta_v;

	int i_tr_low  = this->SearchBin(trans_low);
	int i_tr_high = this->SearchBin(trans_high);

	double alpha_low  = BinLowFraction (trans_low, i_tr_low);
	double alpha_high = BinHighFraction(trans_high, i_tr_high);

	pair_ret.first._alpha = alpha_low;
	pair_ret.first._index = i_tr_low;
	pair_ret.second._alpha = alpha_high;
	pair_ret.second._index = i_tr_high;

	return pair_ret;
}

LifEstimator::~LifEstimator()
{
}

