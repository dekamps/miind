/*
 * CircularDistribution.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#include <MPILib/config.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>


using namespace MPILib::utilities;

CircularDistribution::CircularDistribution() {
}

CircularDistribution::~CircularDistribution() {
}

bool CircularDistribution::isLocalNode(NodeId nodeId) const {
	return getResponsibleProcessor(nodeId) == _mpiProxy.getRank();
}

int CircularDistribution::getResponsibleProcessor(NodeId nodeId) const {
	return nodeId % _mpiProxy.getSize();

}

bool CircularDistribution::isMaster() const {
	return _mpiProxy.getRank() == 0;
}



