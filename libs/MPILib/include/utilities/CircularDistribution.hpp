/*
 * CircularDistribution.hpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_CIRCULARDISTRIBUTION_HPP_
#define MPILIB_UTILITIES_CIRCULARDISTRIBUTION_HPP_

#include <MPILib/include/utilities/NodeDistributionInterface.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>

namespace MPILib {
namespace utilities {

class CircularDistribution: public NodeDistributionInterface {
public:
	CircularDistribution();
	virtual ~CircularDistribution();


	/**
	 * check is a node is local to the processor
	 * @param nodeId The Id of the Node
	 * @return true if the Node is local
	 */
	virtual bool isLocalNode(NodeId nodeId) const;
	/** get the processor number which is responsible for the node
	 * @param nodeId The Id of the Node
	 * @return the processor responsible
	 */
	virtual int getResponsibleProcessor(NodeId nodeId) const;

	/**
	 * If the processor is master (We assume the processor with _processorId=0 is the master)
	 * @return true if the node is the master.
	 */
	virtual bool isMaster() const;
private:
	MPIProxy _mpiProxy;

};

} /* namespace MPILib */
} /* namespace utilities */
#endif /* MPILIB_UTILITIES_CIRCULARDISTRIBUTION_HPP_ */
