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

#ifndef MPILIB_MPINETWORK_HPP_
#define MPILIB_MPINETWORK_HPP_

#include <string>
#include <map>
#include <memory>

#include <MPILib/include/MPINode.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/AlgorithmInterface.hpp>
#include <MPILib/include/NetworkState.hpp>
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>
#include <MPILib/include/NodeType.hpp>
#include <MPILib/include/MPIExternalNode.hpp>

namespace MPILib {

  /** \brief A representation of the network class. Probably the most central class that a client will use. MPINodes and their connections are created through
   * its interface.
   *
   * \section MPINetwork_introduction Introduction to MPINetwork
   *
   * The network can be constructed through its default constructor, and at this stage does not have nodes. Instantiation requires two template arguments: WeightValue
   * and NodeDistribution. WeightValue determines the type of the connections that are used in connecting nodes in the network. Important examples are double,
   * a single real value that indicates the strength of a connection, or DelayedConnection, a type that determines number of connections; efficacy; delay. NodeDistribution
   * is a type that determines how the network simulation will be parallelized by MPI. At present we use CircularDistribution.
   * It is strongly recommended to look at example programs to see how an MPINetwork is instantiated.
   *
   * Nodes can be added to the network through the MPINetwork.addNode method, which takes two arguments: a reference to an AlgorithmInterface, and a NodeType. The method
   * returns an int, which is a handle to the node that has just been created. The AlgorithmInterface
   * determines which algorithm is run during a simulation on a given node. The network is agnostic with regards to what algorithm runs on which node. When a simulation
   * step is made, all the network does is to signal each MPINode that is part of it that it must update its state. The MPINode in turn will forward this command
   * to the algorithm via the AlgorithmInterface. As a consequence MPINetwork can be used for future simulations involving algorithms that currently do not yet exist.
   * The NodeType determines whether a node is excitatory or inhibitory. We find that this
   * prevents errors in providing the sign of the connection. Connectivity values associated with inhibitory nodes must be negative, whilst values associated with
   * excitatory nodes must be positive. This mechanism can be bypassed: for individual nodes by giving them NodeType NEUTRAL, for the network as a whole by
   * MPINetwork.setDalesLaw(false). Connections are inserted between nodes with the aid of the handles that were provide upon node creation.
   * This handle must be converted to a NodeId. The MPINetwork.makeFirstInputOfSecond then can be used to indert a connection between to nodes
   * indicated by two NodeId instances, with a value defined by the WeighValue argument. An MPINetwork must be configured by a SimulationRunParameter, where
   * begin and end time of the simulation are specified, the log and the simulation file names are defined, etc. A simple call to MPINetwork.evolve(), then
   * causes the simulation to be executed.
   *
   *
   *
   * \msc["Node creation"]
   *  client, MPINetwork, "MPINetwork:_localNodes", MPINode;
   *
   *  client     box client [label="client"];
   *  MPINetwork box "MPINetwork:_localNodes" [label="MPINetwork"];
   *  MPINode    box MPINode [label="MPINode"];
   *
   *  client=>MPINetwork[label="addNode(NodeType, const AlgorithmInterface&)", URL="\ref MPINetwork::addNode()"];
   *  MPINetwork=>MPINetwork[label="getMaxNodeId()"];
   *  MPINetwork=>MPINode[label="create(const AlgorithmInterface&)", URL="\ref MPINode::MPINode()"];
   *  MPINetwork<<MPINode[label="returns node"];
   *  MPINetwork=>"MPINetwork:_localNodes"[label="insert(pair<tempNodeId,node>)", URL="\ref MPINetwork::insert()"];
   * \endmsc
   *
   *
   * \section MPINetwork_simulation_loop The Simulation Loop
   *  An MPINetwork instance may live in several threads, if MPI was enabled during compilation.  The MPI coordination requires a boost::communicator object
   * which is part of MPIProxy. The MPINetwork._nodeDistribution keeps track of which NodeId
   * handles are local to the thread. A static map instance - MPINetwork._localNodes - maps NodeId to an MPINode instance. Upon a call to MPINetwork.addNode,
   * it is first established whether the new node lives in the local thread. This is done by a call to the MPINetwork._nodeDistribution object that will
   * work out wether the new node will be local to the current thread or not, based on the new NodeId and its local knowledge of how node numbers relate
   * to processor numbers. If the new NodeId will be current to the local thread, the MPINode constructor is called, providing the new NodeId, a reference
   * to the AlgorithmInterface and NodeType, that were provided by the client upon calling MPINetwork.addNode, and a reference to the MPINetwork._nodeDistribution
   * and the MPINetwork._localNodes distribution. The newly created MPINode will be added to the MPINetwork._localNodes map, with the new NodeId as key.
   * As this map is particular to the address space of the thread, the node has indeed become a local node.
   *
   *\msc["Main loop"]
   *     client, MPINetwork, MPINode, "MPINode:pAlgorithm", Report, ReportHandler, MPIProxy;
   *
   * 	 client box client [label="client"];
   *	 MPINetwork box MPINetwork [label="MPINetwork"];
   *	 MPINode box MPINode [label="MPINode"];
   *	 "MPINode:pAlgorithm" box "MPINode:pAlgorithm" [label="MPINode:pAlgorithm"];
   *	 client note  MPINode [label="SimulationTime < (ReportTime and StateTime)"];
   *
   *	 client=>MPINetwork[label="evolve()"];
   *	 MPINetwork=>MPINode[label="waitAll()"];
   *	 MPINode=>MPIProxy[label="waitAll()"];
   *	 MPINetwork=>MPINode[label="prepareEvolve()"];
   *	 MPINetwork=>MPINode[label="evolve(Time)"];
   *	 MPINode note MPINode [label="CurrentTime < Time"];
   *	 MPINode=>"MPINode:pAlgorithm"[label="evolveNodeState()"];
   *	 MPINode=>MPINode[label="sendOwnActivity()"];
   *	 MPINode=>MPIProxy[label="isend(_node_id,_activity)"];
   *	 MPINode=>MPINode[label="receiveData()"];
   *	 MPINode note MPINode[label="if data not local"];
   *	 MPINode=>MPIProxy[label="irecv(_precursorActivity)"];
   *	 MPINetwork<<MPINode[label="current time"];
   *
   *	 MPINetwork note  MPINode [label="SimulationTime > ReportTime"];
   *	 MPINetwork=>MPINetwork[label="updateReportTime()"];
   *	 MPINetwork=>MPINetwork[label="collectReport(Rate)"];
   *	 MPINetwork=>MPINode[label="reportAll(Rate)"];
   *	 MPINode=>"MPINode:pAlgorithm"[label="getActivity()"];
   *	 MPINode<<"MPINode:pAlgorithm";
   *	 MPINode=>"MPINode:pAlgorithm"[label="getGrid()"];
   *	 MPINode<<"MPINode:pAlgorithm";
   * 	 MPINode=>Report[label="create()"];
   *	 MPINode->ReportHandler[label="writeReport(Report)"];
   *\endmsc
   *
   */
template<class WeightValue, class NodeDistribution>
class MPINetwork{

public:

	typedef WeightValue WeightType;

	explicit MPINetwork();

	~MPINetwork();

	/**
	 * Adds a new node to the network
	 * @param alg The Algorithm of the actual node
	 * @param nodeType The Type of the Node
	 * @return returns the NodeId of the generated node
	 */
	int addNode(const AlgorithmInterface<WeightValue>& alg,
			NodeType nodeType);

	/**
	 * Connect two nodes
	 * @param first NodeId of the first node
	 * @param second NodeId of the second node
	 * @param weight The WeightType of the connection
	 * @exception Can throw a ParallelException
	 */
	void makeFirstInputOfSecond(NodeId first, NodeId second,
			const WeightValue& weight);

	void defineExternalNodeInputAndOutput(NodeId input, NodeId output,
		const WeightValue& in_weight, const WeightValue& out_weight);

	/**
	 * Configure the next simulation
	 * @param simParam The Simulation Parameter
	 */
	void configureSimulation(const SimulationRunParameter& simParam);

	/**
	 * Envolve the network
	 */
	void evolve();

	/**
	 * Envolve the network by a single timestep
	 */
	void evolveSingleStep();

	std::vector<ActivityType> getExternalActivities();

	void setExternalActivities(std::vector<ActivityType> activities);

	void startSimulation();

	void endSimulation();

	MPINode<WeightValue, NodeDistribution>* getNode(NodeId);

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

	void collectReport(report::ReportType type);

	void addExternalNode();

	/**
	 * initialize the log stream
	 * @param filename filename of the log stream
	 */
	void initializeLogStream(const std::string & filename);

	/**
	 * called when the simulation is finished
	 */
	void clearSimulation();

	/**
	 * checks if dales Law is active
	 * @return true if Dales law is set
	 */
	bool isDalesLawSet() const;

	/**
	 * setter for Dales law
	 * @param b_law true if Dales law should be active
	 */
	void setDalesLaw(bool b_law);

	void updateReportTime();
	void updateSimulationTime();
	void updateStateTime();

	/*Time*/ Index getEndTime() const;
	/*Time*/ Index getCurrentReportTime() const;
	/*Time*/ Index getCurrentSimulationTime() const;
	/*Time*/ Index getCurrentStateTime() const;

	static MPIExternalNode<WeightValue, NodeDistribution> _externalNode;

	/**
	 * local nodes of the processor
	 */
	static std::map<NodeId, MPINode<WeightValue, NodeDistribution>> _localNodes;
	/**
	 * The actual distribution of the nodes.
	 */
	static NodeDistribution _nodeDistribution;

	/**
	 * The max Node number assigned so far.
	 * @attention This number is only handled by the master node. Therefore never access it direct!
	 */
	int _maxNodeId = 0;

	MPILib::Index _i_report     = 0;
	MPILib::Index _i_state      = 0;
	MPILib::Index _i_simulation = 0;

	NetworkState _stateNetwork = 0.0;
	bool _isDalesLaw = true;

	std::map<NodeId, NodeType> nodeIdsType_;

	SimulationRunParameter _parameterSimulationRun = SimulationRunParameter(report::handler::InactiveReportHandler(), 0, 0.0,
		0.0, 0.0, 0.0, "");

	unsigned long  _n_sim_steps    = 0;
	MPILib::Number _n_report_steps = 0;
	MPILib::Number _n_state_steps  = 0;

};


//! Standard conversion for operations that depend on efficacy online
inline double toEfficacy(double efficacy) {
	return efficacy;
}

} //end namespace

#endif /* MPILIB_MPINETWORK_HPP_ */
