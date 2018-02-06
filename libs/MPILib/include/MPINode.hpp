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

#ifndef MPILIB_MPINODE_HPP_
#define MPILIB_MPINODE_HPP_

//#include <MPILib/config.hpp>
#include <vector>
#include <map>
#include <memory>

#include <MPILib/include/AlgorithmInterface.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/NodeType.hpp>

#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {

/**
 * \brief Class for nodes in an MPINetwork.
 *
 * An MPINode is responsible for maintaing an algorithm through a shared_ptr to an  AlgorithmInterface object. It is also responsible
 * for being aware of who its predecessors and successors are in the network and thereby inplements the connectivity table. Most of the
 * MPI communication is handled by this class. MPINode::receiveData and MPINode::sendOwnData are used to transmit information about its state, and to
 * collate information about its precursor nodes so that this information can be passed on to its AlgorithmInterface instance.
 * Two hooks are provided for designers of AlgorithmInterface classes:
 * MPINode::prepareEvolve calls AlgorithmInterface::prepareEvolve, which allows algorithms to collate input contributions to this particular node,
 * whilst MPINode::evolve calls AlgorithmInterface::evolve. These methods can then be overloaded by the designers of a sub class of AlgorithmInterface,
 * i.e. the designers of algorithms. The separation between prepareEvolve and evolve ensure that synchronous network updating can be implemented.
 * The MPINetwork::addPrecursor and MPI::Network::addSuccesor are used by MPINetwork::addNode.
 */

template<class Weight, class NodeDistribution>
class MPINode {
public:

	/**
	 * Constructor
	 * @param algorithm Algorithm the algorithm the node should contain
	 * @param nodeType NodeType the type of the node
	 * @param nodeId NodeId the id of the node
	 * @param nodeDistribution The Node Distribution.
	 * @param localNode The local nodes of this processor
	 */
	explicit MPINode(
			const AlgorithmInterface<Weight>& algorithm,
			NodeType nodeType,
			NodeId nodeId,
			const NodeDistribution& nodeDistribution,
			const std::map<NodeId, MPINode<Weight, NodeDistribution>>& localNode,
			const std::string& name = ""
			);

	/**
	 * Destructor
	 */
	virtual ~MPINode();

	/**
	 * Evolve this algorithm over a time
	 * @param time Time until the algorithm should evolve
	 * @return Time the algorithm have evolved, which may be slightly different, due to rounding errors.
	 */
	Time evolve(Time time);

	/**
	 * Called before each evolve call during each evolve. Can
	 * be used to prepare the input for the evolve method.
	 */
	void prepareEvolve();

	/**
	 * Configure the Node with the Simulation Parameters
	 * @param simParam Simulation Parameters
	 */
	void configureSimulationRun(const SimulationRunParameter& simParam);

	/**
	 * Add a precursor to the current node
	 * @param nodeId NodeId the id of the precursor
	 * @param weight the weight of the connection
	 * @param nodeType the nodeType of the precursor
	 */
	void addPrecursor(NodeId nodeId, const Weight& weight, NodeType nodeType);

	/**
	 * Add a successor to the current node
	 * @param nodeId NodeId the id of the successor
	 */
	void addSuccessor(NodeId nodeId);

	/**
	 * Getter for the Nodes activity
	 * @return The current node activity
	 */
	ActivityType getActivity() const;

	/**
	 * The Setter for the node activity
	 * @param activity The activity the node should be in
	 */
	void setActivity(ActivityType activity);

	/**
	 * Receive the new data from the precursor nodes
	 */
	void receiveData();

	/**
	 * Send the own state to the successors.
	 */
	void sendOwnActivity();

	/**
	 * Report the node state
	 * @param type The type of Report
	 */
	void reportAll(report::ReportType type) const;

	/**
	 * finishes the simulation.
	 */
	void clearSimulation();

	/**
	 * returns the type of the node
	 */
	NodeType getNodeType() const;

	/**
	 * Wait that all communication is finished
	 */
	static void waitAll();

	ActivityType getActivity();
	ActivityType getExternalPrecursorActivity();
	void setExternalPrecurserActivity(ActivityType activity);
	void recvExternalPrecurserActivity(NodeId id, int tag);
	void setExternalPrecursor(const Weight& weight, NodeType nodeType);

protected:

	/**
	 * Store the nodeIds of the Precursors
	 */
	std::vector<NodeId> _precursors;

	/**
	 * Store the weights of the connections to the precursors
	 */
	std::vector<Weight> _weights;

	/**
	 * Store the _precursorTypes
	 */
	std::vector<NodeType> _precursorTypes;

	/**
	 * Store the nodeIds of the successors
	 */
	std::vector<NodeId> _successors;

	/**
	 * A Pointer that holds the Algorithm
	 */
	std::shared_ptr<AlgorithmInterface<Weight>> _pAlgorithm;

	/**
	 * The type of this node needed for dales law
	 */
	NodeType _nodeType;

	/**
	 * the Id of this node
	 */
	NodeId _nodeId;

	/**
	 * Reference to the local nodes of the processor. They are owned by the network.
	 */
	const std::map<NodeId, MPINode<Weight, NodeDistribution>>& _rLocalNodes;
	/**
	 * Reference to the NodeDistribution. This is owned by the network.
	 */
	const NodeDistribution& _rNodeDistribution;

	/**
	 *  A node can have a name
	 */
	const std::string _name;
	/**
	 * Activity of this node
	 */
	ActivityType _activity = 0;

	/**
	 * Storage for the state of the precursors, to avoid to much communication.
	 */
	std::vector<ActivityType> _precursorActivity;

	bool _hasExternalPrecursor = false;
	ActivityType _externalPrecursorActivity;
	Weight _externalPrecursorWeight;
	NodeType _externalPrecursorType;

	Number _number_iterations = 0;
	Number _maximum_iterations = 0;

	/**
	 * Pointer to the Report Handler
	 */
	std::shared_ptr<report::handler::AbstractReportHandler> _pHandler;
};

typedef MPINode<double, utilities::CircularDistribution> D_MPINode;

} //end namespace

#endif /* MPILIB_MPINODE_HPP_ */
