/*
 * EmptyAlgorithm.cpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#include <MPILib/include/EmptyAlgorithm.hpp>

namespace MPILib {

EmptyAlgorithm::EmptyAlgorithm(){
	// TODO Auto-generated constructor stub

}

EmptyAlgorithm::~EmptyAlgorithm() {
	// TODO Auto-generated destructor stub
}

EmptyAlgorithm* EmptyAlgorithm::Clone() const {
	return new EmptyAlgorithm(*this);
}

void EmptyAlgorithm::Configure(const SimulationRunParameter& simParam) {

}

void EmptyAlgorithm::EvolveNodeState(const std::vector<NodeState>& nodeVector,
		const std::vector<WeightType>& weightVector, Time time) {

}

} /* namespace MPILib */
