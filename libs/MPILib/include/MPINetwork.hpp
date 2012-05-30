/*
 * MPINetwork.hpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#ifndef MPILIB_MPINETWORK_HPP_
#define MPILIB_MPINETWORK_HPP_

#include <boost/noncopyable.hpp>
#include <string>
#include <map>

#include "MPINode.hpp"
#include "BasicTypes.hpp"

namespace MPILib{

class MPINetwork: private boost::noncopyable {

public:
	MPINetwork();

	~MPINetwork();

	/**
	 * Adds a new node to the network
	 * @param The Algorithm of the actual node
	 * @param The Type of the Node
	 * @return returns the NodeId of the generated node
	 */
	int AddNode(const Algorithm&, NodeType);

	/** Connects two node
	 * @param NodeId of the first node
	 * @param NodeId of the second node
	 * @param The WeightType of the connection
	 * @return A boolean which is true if no error occured
	 * @exception Can throw a ParallelException
	 */
	void MakeFirstInputOfSecond(NodeId, NodeId, const WeightType&);

	//! Configure the next simulation
	bool ConfigureSimulation(const SimulationRunParameter&);

	//! Envolve the network
	void Evolve();

private:
	/** check is a node is local to the processor
	 * @param The Id of the Node
	 * @return true if the Node is local
	 */
	bool isLocalNode(NodeId);
	/** get the processor number which is responsible for the node
	 * @param The Id of the Node
	 * @return the processor responsible
	 */
	int getResponsibleProcessor(NodeId);

	/**
	 * If the processor is master (We assume the processor with _processorId=0 is the master)
	 * @return true if the node is the master.
	 */
	bool isMaster() const;

	/**
	 * returns the max node id currently used.
	 * This is done via a broadcast from the master node.
	 * @return the max node id assigned so far.
	 */
	int getMaxNodeId();

	/**
	 * Increments the NodeId on the master by 1
	 */
	void incrementMaxNodeId();
	/**
	 * local nodes of the processor
	 */
	std::map<NodeId, MPINode> _localNodes;
	/**
	 * The local processor id
	 */
	int _processorId;

	/**
	 * The total number of processors
	 */
	int _totalProcessors;

	/**
	 * The max Node number assigned so far.
	 * @attention This number is only handled by the master node. Therefore never access it direct!
	 */
	int _maxNodeId;

};

}//end namespace

#endif /* MPILIB_MPINETWORK_HPP_ */
