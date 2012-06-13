/*
 * MPINetwork.cpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */
#include <sstream>
#include <iostream>
#include <fstream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/utilities/IterationNumberException.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/reportHandler/InactiveReportHandler.hpp>
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/MPINodeCode.hpp>

namespace mpi = boost::mpi;
using namespace MPILib;

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::MPINetwork() :
		_parameter_simulation_run(InactiveReportHandler(), 0, 0.0,
				0.0, 0.0, 0.0, ""), //
		_stream_log(), //
		_pNodeDistribution(new NodeDistribution), //
		_state_network(0.0), //
		_pLocalNodes(
				new std::map<NodeId, MPINode<WeightValue, NodeDistribution>>) {

	if (_pNodeDistribution->isMaster()) {
		_maxNodeId = 0;
	}
}

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::~MPINetwork() {

}

template<class WeightValue, class NodeDistribution>
int MPINetwork<WeightValue, NodeDistribution>::addNode(
		const algorithm::AlgorithmInterface<WeightValue>& alg,
		NodeType nodeType) {

	int tempNodeId = getMaxNodeId();
	if (_pNodeDistribution->isLocalNode(tempNodeId)) {
		MPINode<WeightValue, NodeDistribution> node = MPINode<WeightValue,
				NodeDistribution>(alg, nodeType, tempNodeId, _pNodeDistribution,
				_pLocalNodes);
		_pLocalNodes->insert(std::make_pair(tempNodeId, node));
	}
	//increment the max NodeId to make sure that it is not assigned twice.
	incrementMaxNodeId();
	return tempNodeId;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::makeFirstInputOfSecond(
		NodeId first, NodeId second, const WeightValue& weight) {

	if (_pNodeDistribution->isLocalNode(first)) {
		if (_pLocalNodes->count(first) > 0) {
			_pLocalNodes->find(first)->second.addSuccessor(second);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << first << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}
	if (_pNodeDistribution->isLocalNode(second)) {
		if (_pLocalNodes->count(second) > 0) {
			_pLocalNodes->find(second)->second.addPrecursor(first, weight);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << second
					<< "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}

}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::configureSimulation(
		const SimulationRunParameter& simParam) {
	_current_report_time = simParam.TReport();
	_current_simulation_time = simParam.TBegin();
	_parameter_simulation_run = simParam;

	initializeLogStream(simParam.LogName());

	try {
		//loop over all local nodes!
		for (auto& it : (*_pLocalNodes)) {
			it.second.configureSimulationRun(simParam);
		}

	} catch (...) {
		_stream_log << "error during configuration/n";
		_stream_log.flush();
	}

	_state_network.ToggleConfigured();

	_stream_log.flush();

}

//! Envolve the network
template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::evolve() {

	std::string report;
	if (_state_network.IsConfigured()) {
		_state_network.ToggleConfigured();
		_stream_log << "Starting simulation\n";
		_stream_log.flush();

		try {
			do {
				do {
					// business as usual: keep evolving, as long as there is nothing to report
					// or to update
					updateSimulationTime();

					//envolve all local nodes
					for (auto& it : (*_pLocalNodes)) {
						it.second.evolve(getCurrentSimulationTime());
					}

				} while (getCurrentSimulationTime() < getCurrentReportTime()
						&& getCurrentSimulationTime() < getCurrentStateTime());

				// now there is something to report or to update
				if (getCurrentSimulationTime() >= getCurrentReportTime()) {
					// there is something to report
					//CheckPercentageAndLog(CurrentSimulationTime());
					updateReportTime();
					report = collectReport(RATE);
					_stream_log << report;
				}
				// just a rate or also a state?
				if (getCurrentSimulationTime() >= getCurrentStateTime()) {
					// a rate as well as a state
					collectReport(STATE);
					updateStateTime();
				}

			} while (getCurrentReportTime() < getEndTime());
			// write out the final state
			collectReport(STATE);
		}

		catch (utilities::IterationNumberException &e) {
			_stream_log << "NUMBER OF ITERATIONS EXCEEDED\n";
			_state_network.SetResult(NUMBER_ITERATIONS_ERROR);
			_stream_log.flush();
			_stream_log.close();
		}

		clearSimulation();
		_stream_log << "Simulation ended, no problems noticed\n";
		_stream_log << "End time: " << getCurrentSimulationTime() << "\n";
		_stream_log.close();
	}

}

template<class WeightValue, class NodeDistribution>
int MPINetwork<WeightValue, NodeDistribution>::getMaxNodeId() {

	mpi::communicator world;
	mpi::broadcast(world, _maxNodeId, 0);
	return _maxNodeId;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::incrementMaxNodeId() {
	if (_pNodeDistribution->isMaster()) {
		_maxNodeId++;
	}
}

template<class WeightValue, class NodeDistribution>
std::string MPINetwork<WeightValue, NodeDistribution>::collectReport(
		ReportType type) {

	std::string string_return;

	for(auto& it: (*_pLocalNodes)) {
		string_return += it.second.reportAll(type);
	}

	return string_return;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::initializeLogStream(
		const std::string & filename) {
	// resource will be passed on to _stream_log
	std::shared_ptr<std::ostream> p_stream(
			new std::ofstream(filename.c_str()));
	if (!p_stream)
		throw utilities::Exception("MPINetwork cannot open log file.");
	if (!_stream_log.OpenStream(p_stream))
		_stream_log << "WARNING YOU ARE TRYING TO REOPEN THIS LOG FILE\n";
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::clearSimulation() {

	for(auto& it: (*_pLocalNodes)) {
		it.second.clearSimulation();
	}

}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateReportTime() {
	_current_report_time += _parameter_simulation_run.TReport();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateSimulationTime() {
	_current_simulation_time += _parameter_simulation_run.TStep();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateStateTime() {
	_current_state_time += _parameter_simulation_run.TState();
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getEndTime() const {
	return _parameter_simulation_run.TEnd();
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getCurrentReportTime() const {
	return _current_report_time;
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getCurrentSimulationTime() const {
	return _current_simulation_time;
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getCurrentStateTime() const {
	return _current_state_time;

}
