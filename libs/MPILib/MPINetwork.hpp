/*
 * MPINetwork.hpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#ifndef MPINETWORK_HPP_
#define MPINETWORK_HPP_

#include <boost/noncopyable.hpp>
#include <string>
#include <vector>

typedef int Algorithm;
typedef int NodeType;
typedef int NodeId;
typedef double WeightType;
typedef int SimulationRunParameter;
typedef int Node;


class MPINetwork: private boost::noncopyable {

public:
	MPINetwork();

	~MPINetwork();

	void AddNode(const Algorithm&, NodeType, NodeId);

	bool MakeFirstInputOfSecond(NodeId, NodeId, WeightType&);

	//! Configure the next simulation
	bool ConfigureSimulation(const SimulationRunParameter&);

	//! Envolve the network
	bool Evolve();


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
	 * local nodes of the processor
	 */
	std::vector<Node> _localNodes;
	/**
	 * The local processor id
	 */
	int _processorId;

	/**
	 * The total number of processors
	 */
	int _totalProcessors;

};

#endif /* MPINETWORK_HPP_ */
