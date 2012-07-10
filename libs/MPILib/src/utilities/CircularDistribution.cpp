/*
 * CircularDistribution.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#include <MPILib/config.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
namespace mpi = boost::mpi;
#endif

using namespace MPILib::utilities;

CircularDistribution::CircularDistribution() {
#ifdef ENABLE_MPI

	mpi::communicator world;

	_processorId = world.rank();
	_totalProcessors = world.size();
#endif
}

CircularDistribution::~CircularDistribution() {
}

bool CircularDistribution::isLocalNode(NodeId nodeId) const {
	return getResponsibleProcessor(nodeId) == _processorId;
}

int CircularDistribution::getResponsibleProcessor(NodeId nodeId) const {
	return nodeId % _totalProcessors;

}

bool CircularDistribution::isMaster() const {
	return _processorId == 0;
}

int CircularDistribution::getRank() const{
	return _processorId;
}

int CircularDistribution::getSize() const{
	return _totalProcessors;
}

