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
#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/HebbianAlgorithm.hpp>
#include <MPILib/include/StringDefinitions.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

#include <NumtoolsLib/NumtoolsLib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

#include <functional>
#include <numeric>

using namespace MPILib;

HebbianAlgorithm::HebbianAlgorithm(const HebbianParameter& par_heb):
_rate(0),
_time(0)
{
}

HebbianAlgorithm::HebbianAlgorithm(const HebbianAlgorithm& rhs):
_rate(rhs._rate),
_time(rhs._time){
}

HebbianAlgorithm::~HebbianAlgorithm(){
}

MPILib::Rate HebbianAlgorithm::getCurrentRate() const {
	return _rate;
}

MPILib::Time HebbianAlgorithm::getCurrentTime() const {
	return _time;
}

MPILib::AlgorithmGrid HebbianAlgorithm::getGrid(NodeId) const {
	vector<MPILib::Efficacy> vec_ret(_rate,1);
	return MPILib::AlgorithmGrid(vec_ret);
}

void HebbianAlgorithm::configure(const SimulationRunParameter& par_run){
	_time = par_run.getTBegin();
}

void HebbianAlgorithm::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {

}


HebbianAlgorithm* HebbianAlgorithm::clone() const {
	return new HebbianAlgorithm(*this);
}
