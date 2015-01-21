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
#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/NetworkState.hpp>
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>
#include <MPILib/include/NodeType.hpp>

namespace MPILib {

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
	int addNode(const algorithm::AlgorithmInterface<WeightValue>& alg,
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

	/**
	 * Configure the next simulation
	 * @param simParam The Simulation Parameter
	 */
	void configureSimulation(const SimulationRunParameter& simParam);

	/**
	 * Envolve the network
	 */
	void evolve();

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

	Time getEndTime() const;
	Time getCurrentReportTime() const;
	Time getCurrentSimulationTime() const;
	Time getCurrentStateTime() const;

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

	Time _currentReportTime = 0;
	Time _currentStateTime = 0;
	Time _currentSimulationTime = 0;
	NetworkState _stateNetwork = 0.0;
	bool _isDalesLaw = true;

	std::map<NodeId, NodeType> nodeIdsType_;

	SimulationRunParameter _parameterSimulationRun = SimulationRunParameter(report::handler::InactiveReportHandler(), 0, 0.0,
		0.0, 0.0, 0.0, "");

};


//! Standard conversion for operations that depend on efficacy online
inline double toEfficacy(double efficacy) {
	return efficacy;
}

} //end namespace

#endif /* MPILIB_MPINETWORK_HPP_ */
