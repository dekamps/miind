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
#include <cmath>
#include <vector>
#include "LifNeuralDynamics.hpp"

using namespace GeomLib;

LifNeuralDynamics::LifNeuralDynamics
(
	const OdeParameter& par,
	double lambda
):
AbstractNeuralDynamics(par),
_lambda(lambda),
_t_period(this->TimePeriod()),
_N_pos(Nposinit()),
_t_step(this->TStep()),
_N_neg(Nneginit())
{
}


LifNeuralDynamics::~LifNeuralDynamics()
{
}


MPILib::Potential LifNeuralDynamics::EvolvePotential
(
	MPILib::Potential V_0,
	MPILib::Time t
) const
{
	return _par._par_pop._V_reversal + exp(-t/_par._par_pop._tau)*(V_0 - _par._par_pop._V_reversal);
}


LifNeuralDynamics* LifNeuralDynamics::Clone() const {
  return new LifNeuralDynamics(*this);
}

Number LifNeuralDynamics::Nneginit() const
{
  float n_extra = (_par._par_pop._tau/_t_step)*log((_par._par_pop._theta - _par._par_pop._V_reversal)/(_par._par_pop._V_reversal - _par._V_min));
  Number N_min = static_cast<Number>(ceil(_N_pos - (n_extra)));
  return N_min;
}

Number LifNeuralDynamics::Nposinit() const
{
  return _par._nr_bins;
}


MPILib::Time LifNeuralDynamics::TStep() const
{
  return _t_period/(_N_pos - 1);
}

MPILib::Time LifNeuralDynamics::TimePeriod() const
{
  assert( _par._V_min   <= _par._par_pop._V_reset);
  assert( _par._par_pop._V_reset <  _par._par_pop._theta );

  double V_plus = _par._par_pop._V_reversal + _lambda*(_par._par_pop._theta - _par._par_pop._V_reversal);
  return _par._par_pop._tau*log((_par._par_pop._theta -_par._par_pop._V_reversal)/(V_plus - _par._par_pop._V_reversal));
}


std::vector<MPILib::Potential> LifNeuralDynamics::InterpretationArray() const
{
  std::vector<MPILib::Potential> vec_ret(_N_pos +_N_neg);
  assert(_N_pos + _N_neg > 3);

  vec_ret[ 0 + _N_neg] = _par._par_pop._V_reversal;
  vec_ret[ 1 + _N_neg] = _par._par_pop._V_reversal + _lambda*(_par._par_pop._theta - _par._par_pop._V_reversal);

  for (MPILib::Index i = 2; i < _N_pos; i++)
	  vec_ret[i + _N_neg] = _par._par_pop._V_reversal + (_par._par_pop._theta - _par._par_pop._V_reversal)*exp((-_t_step/_par._par_pop._tau)*(_N_pos - i));

  if (_N_neg > 0){
	  vec_ret[-1 + _N_neg] = _par._par_pop._V_reversal - _lambda*(_par._par_pop._theta- _par._par_pop._V_reversal);

	  for (int i = -2; i >= - static_cast<int>(_N_neg); i--)
		  vec_ret[i + _N_neg] = _par._par_pop._V_reversal - (_par._par_pop._theta - _par._par_pop._V_reversal)*exp((-_t_step/_par._par_pop._tau)*((int)_N_pos + i));
  }
  return vec_ret;
}

