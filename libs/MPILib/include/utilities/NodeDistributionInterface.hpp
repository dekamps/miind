/*
 * MPIDistributionInterface.hpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_NODEDISTRIBUTIONINTERFACE_HPP_
#define MPILIB_UTILITIES_NODEDISTRIBUTIONINTERFACE_HPP_

#include <boost/noncopyable.hpp>
#include "../BasicTypes.hpp"


namespace MPILib{
namespace utilities{


/**
 * @class NodeDistributionInterface abstract interface for implementing concrete distributions
 */
class NodeDistributionInterface: private boost::noncopyable{
public:
	NodeDistributionInterface(){};

	virtual ~NodeDistributionInterface(){};

	/** check is a node is local to the processor
	 * @param The Id of the Node
	 * @return true if the Node is local
	 */
	virtual bool isLocalNode(NodeId nodeId) const= 0;
	/** get the processor number which is responsible for the node
	 * @param The Id of the Node
	 * @return the processor responsible
	 */
	virtual int getResponsibleProcessor(NodeId nodeId) const= 0;

	/**
	 * If the processor is master (We assume the processor with _processorId=0 is the master)
	 * @return true if the node is the master.
	 */
	virtual bool isMaster() const = 0;

	/**
	 * Get the rank of the processor
	 * @return the rank of the processor
	 */
	virtual int getRank() const = 0;

	/**
	 * Get the size of the system
	 * @return the size of the system
	 */
	virtual int getSize() const = 0;


};
}//end namespace
}//end namespace

#endif /* MPILIB_UTILITIES_NODEDISTRIBUTIONINTERFACE_HPP_ */
