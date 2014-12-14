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
#include "SpikingLifNeuralDynamics.hpp"

using namespace GeomLib;
using  MPILib::Time;

SpikingLifNeuralDynamics::SpikingLifNeuralDynamics
(
		const OdeParameter& par_ode,
		const LIFParameter& par_lif
):
SpikingNeuralDynamics(par_ode),
_par_lif(par_lif)
{
}

SpikingLifNeuralDynamics::SpikingLifNeuralDynamics
(
	const SpikingLifNeuralDynamics& rhs
):
SpikingNeuralDynamics(rhs),
_par_lif(rhs._par_lif)
{
}

SpikingLifNeuralDynamics::~SpikingLifNeuralDynamics()
{
}

Potential SpikingLifNeuralDynamics::EvolvePotential(Potential V, Time t) const
{
	// evolve the potential under the assumption that no crossing to infinity
	// takes places within time t

	assert(this->TimeToInf(V) > t);

	double V_ret = (_par_lif.CurrentCompensation() + _par._par_pop._V_reversal) - ((_par_lif.CurrentCompensation()+_par._par_pop._V_reversal) -V)*exp(-t/_par._par_pop._tau);
	return V_ret;
}

Time SpikingLifNeuralDynamics::TimeToInf(Potential V) const
{
	double t = (this->_par._par_pop._tau)*log( (_par_lif.CurrentCompensation() + _par._par_pop._V_reversal- V)/(_par_lif.CurrentCompensation() + _par._par_pop._V_reversal - _par._par_pop._theta));
	return t;
}

SpikingLifNeuralDynamics* SpikingLifNeuralDynamics::Clone() const
{
	return new SpikingLifNeuralDynamics(*this);
}

Time SpikingLifNeuralDynamics::TPeriod() const
{
	return this->TimeToInf(_par._V_min);
}


Time SpikingLifNeuralDynamics::TStep() const
{
	return this->TPeriod()/_par._nr_bins;
}



