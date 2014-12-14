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
#include "SpikingQifNeuralDynamics.hpp"

using namespace GeomLib;

SpikingQifNeuralDynamics::SpikingQifNeuralDynamics
(
	const OdeParameter& par_ode,
	const QifParameter& par_qif
):
SpikingNeuralDynamics
(
	par_ode
),
_par_qif(par_qif)
{
}

SpikingQifNeuralDynamics::SpikingQifNeuralDynamics
(
	const SpikingQifNeuralDynamics& dyn
):
SpikingNeuralDynamics(dyn),
_par_qif(dyn._par_qif)
{
}

SpikingQifNeuralDynamics::~SpikingQifNeuralDynamics()
{
}

Potential SpikingQifNeuralDynamics::EvolvePotential(Potential V, Time t) const
{
	// evolve the potential under the assumption that no crossing to infinity
	// takes places within time t
	assert(this->TimeToInf(V) > t);

	double sqr = sqrt(_par_qif.Gammasys());
	Potential V_ret = sqr*tan(sqr*(t/this->_par._par_pop._tau) + atan(V/sqr));
	return V_ret;
}

Time SpikingQifNeuralDynamics::TimeToInf(Potential V) const
{
	double sqr = sqrt(_par_qif.Gammasys());
	double t = (this->_par._par_pop._tau/sqr)*(atan(this->_par._par_pop._theta/sqr) - atan(V/sqr));
	return t;
}

SpikingQifNeuralDynamics* SpikingQifNeuralDynamics::Clone() const
{
	return new SpikingQifNeuralDynamics(*this);
}


Time SpikingQifNeuralDynamics::TPeriod() const
{
	return this->TimeToInf(_par._V_min);
}


Time SpikingQifNeuralDynamics::TStep() const
{
	return this->TPeriod()/_par._nr_bins;
}
