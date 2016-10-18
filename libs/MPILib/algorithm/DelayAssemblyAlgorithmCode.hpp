// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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

#ifndef MPILIB_ALGORITHMS_DELAYASSEMBLYALGORITHM_CODE_HPP_
#define MPILIB_ALGORITHMS_DELAYASSEMBLYALGORITHM_CODE_HPP_

#include <cassert>
#include <MPILib/algorithm/DelayAssemblyAlgorithm.hpp>
#include "MuSigmaScalarProduct.hpp"

namespace MPILib {
namespace algorithm {

    template <class WeightType>
    DelayAssemblyAlgorithm<WeightType>::DelayAssemblyAlgorithm(const DelayAssemblyParameter& par):
    _par(par),
    _t_current(0.0),
    _r_current(0.0)
    {
    }

    template <class WeightType>
    DelayAssemblyAlgorithm<WeightType>::~DelayAssemblyAlgorithm()
    {
    }

    template <class WeightType>
    void DelayAssemblyAlgorithm<WeightType>::configure(const SimulationRunParameter& par_run)
    {
    }

    template <class WeightType>
    DelayAssemblyAlgorithm<WeightType>* DelayAssemblyAlgorithm<WeightType>::clone() const {
    	return new DelayAssemblyAlgorithm<WeightType>(*this);
    }

    template <class WeightType>
    AlgorithmGrid DelayAssemblyAlgorithm<WeightType>::getGrid(MPILib::NodeId) const {
    	vector<double> g(1);
    	g[0] = _r_current;
    	return AlgorithmGrid(g);
    }

    template <class WeightType>
	void DelayAssemblyAlgorithm<WeightType>::evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<WeightType>& weightVector, Time time){
    	_t_current = time;

    	GeomLib::MuSigmaScalarProduct<WeightType> prod;
    	// don't use the membrane time constant; not interested in diffusion approximation
    	GeomLib::MuSigma ms = prod.Evaluate(nodeVector,weightVector,1.0);

    	if (ms._mu > _par._th_exc)
    		_r_current = _par._rate;

    	if (ms._mu < _par._th_inh)
    		_r_current = 0.0;


    }


} /* end namespace algorithm */
} /* end namespace MPILib */

#endif  //include guard MPILIB_ALGORITHMS_DELAYASSEMBLYALGORITHM_CODE_HPP_
