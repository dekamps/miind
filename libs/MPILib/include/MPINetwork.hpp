/*
 * MPINetwork.hpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#ifndef MPILIB_MPINETWORK_HPP_
#define MPILIB_MPINETWORK_HPP_

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <string>
#include <map>

#include "MPINode.hpp"
#include "BasicTypes.hpp"
#include "utilities/NodeDistributionInterface.hpp"


namespace MPILib{

class MPINetwork: private boost::noncopyable {

public:
	explicit MPINetwork();

	~MPINetwork();

	/**
	 * Adds a new node to the network
	 * @param alg The Algorithm of the actual node
	 * @param nodeType The Type of the Node
	 * @return returns the NodeId of the generated node
	 */
	int AddNode(const Algorithm& alg, NodeType nodeType);

	/**
	 * Connects two node
	 * @param first NodeId of the first node
	 * @param second NodeId of the second node
	 * @param weight The WeightType of the connection
	 * @exception Can throw a ParallelException
	 */
	void MakeFirstInputOfSecond(NodeId first, NodeId second, const WeightType& weight);

	/**
	 * Configure the next simulation
	 * @param simParam The Simulation Parameter
	 */
	void ConfigureSimulation(const SimulationRunParameter& simParam);

	/**
	 * Envolve the network
	 */
	void Evolve();

private:

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
	 * Shared pointer to the actual distribution of the nodes.
	 */
	boost::shared_ptr<utilities::NodeDistributionInterface> _nodeDistribution;

	/**
	 * local nodes of the processor
	 */
	std::map<NodeId, MPINode> _localNodes;


	/**
	 * The max Node number assigned so far.
	 * @attention This number is only handled by the master node. Therefore never access it direct!
	 */
	int _maxNodeId;

};

}//end namespace

#endif /* MPILIB_MPINETWORK_HPP_ */
