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
#include <MPILib/include/PersistantAlgorithm.hpp>
#include <MPILib/include/StringDefinitions.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

#include <NumtoolsLib/NumtoolsLib.h>


#include <functional>
#include <numeric>



namespace MPILib {
namespace algorithm {

PersistantAlgorithm::PersistantAlgorithm():
_rate(0.0),
_time(0.0)
{
}

PersistantAlgorithm::PersistantAlgorithm(const PersistantAlgorithm& rhs) :
_rate(rhs._rate),
_time(rhs._time)
{
}

PersistantAlgorithm::~PersistantAlgorithm() {
}

PersistantAlgorithm* PersistantAlgorithm::clone() const {
	return new PersistantAlgorithm(*this);
}

void PersistantAlgorithm::configure(const SimulationRunParameter& simParam) {

	_time = simParam.getTBegin();
}

void PersistantAlgorithm::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {

	double f_inner_product = innerProduct(nodeVector, weightVector);
	_time = time;

	if (_rate == 0.0 && f_inner_product >  2.0)
		_rate = 20.0;
	if (_rate >  0.0 && f_inner_product >= 0.0)
		_rate = 20.0;
	if ( f_inner_product < 0.0)
		_rate = 0.0;

}

Time PersistantAlgorithm::getCurrentTime() const {
  return _time;
}

Rate PersistantAlgorithm::getCurrentRate() const {
  return _rate;
}

double PersistantAlgorithm::innerProduct(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector) {

	assert(nodeVector.size()==weightVector.size());

	if (nodeVector.begin() == nodeVector.end())
		return 0;

	return std::inner_product(nodeVector.begin(), nodeVector.end(),
			weightVector.begin(), 0.0);

}

std::vector<double> PersistantAlgorithm::getInitialState() const {
	std::vector<double> array_return(WILSON_COWAN_STATE_DIMENSION);
	array_return[0] = 0;
	return array_return;
}

AlgorithmGrid PersistantAlgorithm::getGrid(NodeId, bool) const {
	std::vector<double> array_return(WILSON_COWAN_STATE_DIMENSION);
	array_return[0] = 0;
	return AlgorithmGrid(array_return);
}

} /* namespace algorithm */
} /* namespace MPILib */
