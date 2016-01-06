// Copyright (c) 2005 - 2012 Marc de Kamps
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
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <GeomLib/BinEstimator.hpp>
#include "../UtilLib/UtilLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "GeomLibException.hpp"

using namespace GeomLib;
using namespace NumtoolsLib;
using namespace UtilLib;

BinEstimator::BinEstimator
(
	const vector<double>& vec_interpretation,
	const OdeParameter& par_ode
):
_par_ode(par_ode),
_vec_interpretation(vec_interpretation)
{
}

BinEstimator::~BinEstimator(){
}

Index BinEstimator::SearchBin(Potential v) const {
	if( v > _par_ode._par_pop._theta )
		throw GeomLibException("BinEstimator: v larger than V_max");
	if( v < _par_ode._V_min )
		throw GeomLibException("BinEstimator: v smaller than V_min");
	Number n = _vec_interpretation.size();
	if ( v > _vec_interpretation.back() )
		return n-1;
	// changed search: now from one onwards
	for (Index i = 1; i < n; i++)
		if ( v < _vec_interpretation[i] )
			return i-1;

	throw GeomLibException("No sensible return for SearchBin");
		
}

int BinEstimator::Search(Index ind, Potential v, Potential dv) const {
  int sg = (dv > 0) - (dv < 0);
  int n = static_cast<int>(_vec_interpretation.size());
  for (Index i = 0; i < n; i++){
    Index j = modulo(ind + sg*i, n);
    if (_vec_interpretation[j] <= v && (j == n-1 || _vec_interpretation[j+1] > v) )
      return j;
  }
  assert(false);
  throw GeomLibException("BinEstimator search failed");
}

int BinEstimator::SearchBin(Index ind, Potential v, Potential dv) const {

	int i = ind;
	if ( v < _par_ode._V_min)
		return -1;
	int n = static_cast<int>(_vec_interpretation.size());
	if ( v > _par_ode._par_pop._theta)
		return n;

	if (dv == 0)
		return i;

	int i_ret = Search(i,v, dv);
	return i_ret;			
}

Potential BinEstimator::Translate(Potential v, Potential delta_v) const
{
	Potential diff  = v + delta_v;
	return diff;
}

double BinEstimator::BinLowFraction(Potential v, int i_tr_low) const {
	int n = _vec_interpretation.size();
	if (i_tr_low < 0 || i_tr_low > n-1)
		return 0.0;

	Potential low_bin  = _vec_interpretation[i_tr_low];
	Potential high_bin = (i_tr_low != static_cast<int>(_vec_interpretation.size()) - 1)? _vec_interpretation[i_tr_low + 1] : _par_ode._par_pop._theta;
	assert( low_bin <= v && high_bin >= v);
	return  (high_bin -v )/(high_bin - low_bin);
}

double BinEstimator::BinHighFraction(Potential v, int i_tr_high) const {
	int n = _vec_interpretation.size();
	if (i_tr_high < 0 || i_tr_high > n-1)
		return 0.0;

	Potential low_bin  = _vec_interpretation[i_tr_high];
	Potential high_bin = (i_tr_high != static_cast<int>(_vec_interpretation.size()) - 1)? _vec_interpretation[i_tr_high + 1] : _par_ode._par_pop._theta;
	assert( low_bin <= v && high_bin >= v);
	return (v - low_bin)/(high_bin - low_bin);
}

BinEstimator::CoverPair BinEstimator::CalculateBinCover(Index i, Potential delta_v) const{
	assert(i < _vec_interpretation.size());
	CoverPair pair_ret;
	Potential low = _vec_interpretation[i];
	Potential high = (i != _vec_interpretation.size() - 1) ? _vec_interpretation[i+1] : _par_ode._par_pop._theta;

	Potential trans_low  = Translate(low,	delta_v);
	Potential trans_high = Translate(high,	delta_v);

	int i_tr_low  = this->SearchBin(i,trans_low, delta_v);
	int i_tr_high = this->SearchBin(i,trans_high,delta_v);

	double alpha_low  = BinLowFraction (trans_low, i_tr_low);
	double alpha_high = BinHighFraction(trans_high, i_tr_high);

	pair_ret.first._alpha = alpha_low;
	pair_ret.first._index = i_tr_low;
	pair_ret.second._alpha = alpha_high;
	pair_ret.second._index = i_tr_high;

	return pair_ret;
}
