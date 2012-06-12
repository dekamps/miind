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

#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/MPINode.hpp>

#include <DynamicLib/InactiveReportHandler.h>
#include <DynamicLib/NetworkState.h>

namespace MPILib {

template<class WeightValue, class NodeDistribution>
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
	int addNode(const algorithm::AlgorithmInterface<WeightValue>& alg, NodeType nodeType);

	/**
	 * Connects two node
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
	void configureSimulation(
			const DynamicLib::SimulationRunParameter& simParam);

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

	std::string collectReport(DynamicLib::ReportType type);

	/**
	 * initialize the log stream
	 * @param filename filename of the log stream
	 */
	void initializeLogStream(const std::string & filename);

	/**
	 * called when the simulation is finished
	 */
	void clearSimulation();


	void updateReportTime				();
	void updateSimulationTime			();
	void updateUpdateTime				();
	void updateStateTime				();

	Time getEndTime						() const;
	Time getCurrentReportTime			() const;
	Time getCurrentSimulationTime		() const;
	Time getCurrentUpdateTime			() const;
	Time getCurrentStateTime			() const;

	/**
	 * Shared pointer to the actual distribution of the nodes.
	 */
	boost::shared_ptr<NodeDistribution> _pNodeDistribution;

	/**
	 * local nodes of the processor
	 */
	boost::shared_ptr<std::map<NodeId, MPINode<WeightValue, NodeDistribution>>> _pLocalNodes;

	/**
	 * The max Node number assigned so far.
	 * @attention This number is only handled by the master node. Therefore never access it direct!
	 */
	int _maxNodeId;

	Time _current_report_time = 0;
	Time _current_update_time = 0;
	Time _current_state_time = 0;
	Time _current_simulation_time = 0;
	DynamicLib::NetworkState _state_network;

	DynamicLib::SimulationRunParameter _parameter_simulation_run;
	UtilLib::LogStream _stream_log;

};

} //end namespace

#endif /* MPILIB_MPINETWORK_HPP_ */
