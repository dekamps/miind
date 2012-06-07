/*
 * MPINetwork.cpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */
#include <sstream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/MPINodeCode.hpp>
#include <DynamicLib/IterationNumberException.h>

namespace mpi = boost::mpi;
using namespace MPILib;

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::MPINetwork() :

		_current_report_time(0), _current_update_time(0), _current_state_time(
				0), _current_simulation_time(0), _parameter_simulation_run(
				DynamicLib::InactiveReportHandler(), 0, 0.0, 0.0, 0.0, 0.0, 0.0,
				""), _stream_log(), _pNodeDistribution(new NodeDistribution), _state_network(
				0.0), _pLocalNodes(
				new std::map<NodeId, MPINode<WeightValue, NodeDistribution> >) {

	if (_pNodeDistribution->isMaster()) {
		_maxNodeId = 0;
	}

	//set to one as no node should exist at this time point

}

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::~MPINetwork() {

}

template<class WeightValue, class NodeDistribution>
int MPINetwork<WeightValue, NodeDistribution>::AddNode(
		const AlgorithmInterface<WeightValue>& alg, NodeType nodeType) {

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
void MPINetwork<WeightValue, NodeDistribution>::MakeFirstInputOfSecond(
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
void MPINetwork<WeightValue, NodeDistribution>::ConfigureSimulation(
		const DynamicLib::SimulationRunParameter& simParam) {
	_current_report_time = simParam.TReport();
	_current_update_time = simParam.TUpdate();
	_current_simulation_time = simParam.TBegin();

	_parameter_simulation_run = simParam;

	//loop over all local nodes!
	typename std::map<NodeId, MPINode<WeightValue, NodeDistribution> >::iterator it;
	for (it = _pLocalNodes->begin(); it != _pLocalNodes->end(); it++) {
		it->second.ConfigureSimulationRun(simParam);
	}

}

//! Envolve the network
template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::Evolve() {

	if (_state_network.IsConfigured()) {
		_state_network.ToggleConfigured();
		_stream_log << "Starting simulation\n";
		_stream_log.flush();

		try {
			do {
				do {
					// business as usual: keep evolving, as long as there is nothing to report
					// or to update
					UpdateSimulationTime();

					//envolve all local nodes
					for (std::map<NodeId, D_MPINode>::iterator it =
							_pLocalNodes->begin(); it != _pLocalNodes->end();
							it++) {
						std::cout << "processorID:\t"
								<< _pNodeDistribution->getRank();
						//FIXME change to better time
						it->second.Evolve(1);
						std::cout << std::endl;
						//TODO call evolve on the nodes
					}

				} while (CurrentSimulationTime() < CurrentReportTime()
						&& CurrentSimulationTime() < CurrentUpdateTime()
						&& CurrentSimulationTime() < CurrentStateTime());

				// now there is something to report or to update
				if (CurrentSimulationTime() >= CurrentReportTime()) {
					// there is something to report
					//CheckPercentageAndLog(CurrentSimulationTime());
					UpdateReportTime();
					//report = _implementation.CollectReport(RATE);
					_stream_log << "blub";
				}
				// just a rate or also a state?
				if (CurrentSimulationTime() >= CurrentStateTime()) {
					// a rate as well as a state
					//_implementation.CollectReport(STATE);
					UpdateStateTime();
				}
				// update?
				if (CurrentReportTime() >= CurrentUpdateTime()) {
					//_implementation.CollectReport(UPDATE);
					UpdateUpdateTime();
				}
			} while (CurrentReportTime() < EndTime());
			// write out the final state
			//_implementation.CollectReport(STATE);
		}

		catch (DynamicLib::IterationNumberException &e) {
			_stream_log << "NUMBER OF ITERATIONS EXCEEDED\n";
			_state_network.SetResult(DynamicLib::NUMBER_ITERATIONS_ERROR);
			_stream_log.flush();
			_stream_log.close();
		}

		//_implementation.ClearSimulation();
		_stream_log << "Simulation ended, no problems noticed\n";
		_stream_log << "End time: " << CurrentSimulationTime() << "\n";
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
void MPINetwork<WeightValue, NodeDistribution>::UpdateReportTime() {
	_current_report_time += _parameter_simulation_run.TReport();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::UpdateSimulationTime() {
	_current_simulation_time += _parameter_simulation_run.TStep();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::UpdateUpdateTime() {
	_current_update_time += _parameter_simulation_run.TUpdate();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::UpdateStateTime() {
	_current_state_time += _parameter_simulation_run.TState();
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::EndTime() const {
	return _parameter_simulation_run.TEnd();
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::CurrentReportTime() const {
	return _current_report_time;
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::CurrentSimulationTime() const {
	return _current_simulation_time;
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::CurrentUpdateTime() const {
	return _current_update_time;
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::CurrentStateTime() const {
	return _current_state_time;

}
