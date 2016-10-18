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
#include "../NumtoolsLib/NumtoolsLib.h"
#include "InputParameterSet.hpp"
#include "MuSigma.hpp"
#include "QifOdeSystem.hpp"
#include "GeomLibException.hpp"

using namespace GeomLib;

QifOdeSystem::QifOdeSystem
(
	const SpikingQifNeuralDynamics& dyn
):
SpikingOdeSystem
(
	dyn
),
_par_qif(dyn.ParQif())
{
  _t_current = 0.0;
}

QifOdeSystem::QifOdeSystem(const QifOdeSystem& sys):
SpikingOdeSystem		
(
 sys
),
_par_qif(sys._par_qif)
{
}

QifOdeSystem::~QifOdeSystem()
{
}

void QifOdeSystem::Evolve
(
	Time t
)
{
  if ( ! NumtoolsLib::IsApproximatelyEqualTo(t, _t_step, 1e-9 ) )
    throw GeomLibException("QIFOdeSystem is designed to only have one fixed time step");

  StoreInQueue();
  RetrieveFromQueue();

  this->UpdateIndex();
  this->UpdateCacheMap();
 _t_current += t;
}

QifOdeSystem* QifOdeSystem::Clone() const
{
	return new QifOdeSystem(*this);
}

MPILib::Rate QifOdeSystem::CurrentRate() const
{
  MPILib::Index i_th;
	Number n_bins = this->NumberOfBins();
	i_th = (_index < 0 ) ? _index + n_bins : _index;

	MPILib::Probability prob = _buffer_mass[i_th];

	return prob/(_t_step);
}
