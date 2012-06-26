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
#include <memory>

#include <MPILib/include/MPINode.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/NetworkState.hpp>
#include <MPILib/include/utilities/LogStream.hpp>
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>
#include <MPILib/include/NodeType.hpp>


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
	int addNode(const algorithm::AlgorithmInterface<WeightValue>& alg,
			NodeType nodeType);

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

	std::string collectReport(report::ReportType type);

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
	std::shared_ptr<std::map<NodeId, MPINode<WeightValue, NodeDistribution>>>_pLocalNodes {new std::map<NodeId, MPINode<WeightValue, NodeDistribution>>};

	/**
	 * Shared pointer to the actual distribution of the nodes.
	 */
	std::shared_ptr<NodeDistribution> _pNodeDistribution { new NodeDistribution };

	/**
	 * The max Node number assigned so far.
	 * @attention This number is only handled by the master node. Therefore never access it direct!
	 */
	int _maxNodeId {0};

	Time _currentReportTime {0};
	Time _currentStateTime {0};
	Time _currentSimulationTime {0};
	NetworkState _stateNetwork {0.0};
	bool _isDalesLaw {true};

	SimulationRunParameter _parameterSimulationRun {report::handler::InactiveReportHandler(), 0, 0.0,
		0.0, 0.0, 0.0, ""};
	utilities::LogStream _streamLog {};



};

//! Standard conversion for operations that depend on efficacy online
inline double toEfficacy( double efficacy ) { return efficacy; }

} //end namespace

#endif /* MPILIB_MPINETWORK_HPP_ */
