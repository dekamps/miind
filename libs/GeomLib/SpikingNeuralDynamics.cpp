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
#include "SpikingNeuralDynamics.hpp"

using namespace GeomLib;
using namespace MPILib;

SpikingNeuralDynamics::SpikingNeuralDynamics
(
	const OdeParameter& par_ode
):
AbstractNeuralDynamics(par_ode)
{
}

SpikingNeuralDynamics::SpikingNeuralDynamics
(
	const SpikingNeuralDynamics& rhs
):
AbstractNeuralDynamics(rhs)
{
}

SpikingNeuralDynamics::~SpikingNeuralDynamics()
{
}

std::vector<Potential> SpikingNeuralDynamics::InterpretationArray() const
{
	Number n_bins = this->Par()._nr_bins;

	std::vector<Potential> vec_ret(n_bins);

	Time delta_t = this->TPeriod()/n_bins;
	for (Index i = 0; i < n_bins; i++)
		vec_ret[i] = this->EvolvePotential(this->_par._V_min,i*delta_t);

	return vec_ret;
}
