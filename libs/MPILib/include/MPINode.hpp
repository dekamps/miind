/*
 * MPINode.h
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#ifndef MPILIB_MPINODE_HPP_
#define MPILIB_MPINODE_HPP_

#include <boost/noncopyable.hpp>
#include <vector>
#include <boost/mpi/request.hpp>

#include "BasicTypes.hpp"

namespace MPILib {

/**
 * @class MPINode the class for the actual network nodes. T
 */
class MPINode {
public:
	/**
	 * Constructor
	 * @param Algorithm the algorithm the node should contain
	 * @param NodeType the type of the node
	 */
	explicit MPINode(const Algorithm&, NodeType, NodeId);

	/**
	 * Destructor
	 */
	~MPINode();

	/**
	 * Evolve this algorithm over a time
	 * @param Time until the algorithm should evolve
	 * @return Time the algorithm have evolved
	 */
	Time Evolve(Time);

	/**
	 * Configure the Node with the Simulation Parameters
	 * @param Simulation Parameters
	 * @return true if it worked correct
	 */
	bool ConfigureSimulationRun(const SimulationRunParameter&);

	/**
	 * Add a precursor to the current node
	 * @param NodeId the id of the precursor
	 * @param WeightType the weight of the connection
	 */
	void addPrecursor(NodeId, const WeightType&);

	/**
	 * Add a successor to the current node
	 * @param NodeId the id of the successor
	 * @param WeightType the weight of the connection
	 */
	void addSuccessor(NodeId, const WeightType&);

	/**
	 * Getter for the Nodes state
	 * @return The current node state
	 */
	NodeState getState() const;

	/**
	 * The Setter for the node state
	 * @param state The state the node should be in
	 */
	void setState(NodeState state);

	/**
	 * Receive the new data from the precursor nodes
	 */
	void receiveData();

	/**
	 * Send the own state to the successors.
	 */
	void sendOwnState();

private:

	/** get the processor number which is responsible for the node
	 * @param The Id of the Node
	 * @return the processor responsible
	 */
	int getResponsibleProcessor(NodeId);

	void waitAll();

	std::vector<NodeId> _precursors;

	std::vector<WeightType> _weights;

	std::vector<NodeId> _successors;

	Algorithm _algorithm;

	NodeType _nodeType;

	/**
	 * the Id of this node
	 */
	NodeId _nodeId;



	/**
	 * The local processor id
	 */
	int _processorId;

	/**
	 * The total number of processors
	 */
	int _totalProcessors;

	/**
	 * The state of the node it is currently
	 */
	NodeState _state;

	/**
	 * Storage for the state of the precursors, to avoid to much communication.
	 */
	std::vector<NodeState> _precursorStates;

	std::vector<boost::mpi::request> _mpiStatus;
};

} //end namespace

#endif /* MPILIB_MPINODE_HPP_ */
