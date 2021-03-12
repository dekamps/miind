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

#ifndef MPILIB_MPINETWORK_CODE_HPP_
#define MPILIB_MPINETWORK_CODE_HPP_

//#include <MPILib/config.hpp>
#include <sstream>
#include <iostream>
#include <cassert>
#include <fstream>
#include <boost/timer/timer.hpp>
#include <MPILib/include/utilities/Log.hpp>
#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/utilities/IterationNumberException.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/MPINodeCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>
#include <MPILib/include/utilities/Log.hpp>

namespace MPILib {

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::MPINetwork(){
}

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::~MPINetwork() {
}

template<class WeightValue, class NodeDistribution>
int MPINetwork<WeightValue, NodeDistribution>::addNode(
		const AlgorithmInterface<WeightValue>& alg,
		NodeType nodeType) {

	assert(
			nodeType == EXCITATORY_GAUSSIAN ||
			nodeType == INHIBITORY_GAUSSIAN ||
			nodeType == NEUTRAL ||
			nodeType == EXCITATORY_DIRECT ||
			nodeType == INHIBITORY_DIRECT);

	NodeId tempNodeId = getMaxNodeId();
	if (_nodeDistribution.isLocalNode(tempNodeId)) {
		MPINode<WeightValue, NodeDistribution> node = MPINode<WeightValue,
				NodeDistribution>(alg, nodeType, tempNodeId, _nodeDistribution,
				_localNodes);
		_localNodes.insert(std::make_pair(tempNodeId, node));
		LOG(utilities::logDEBUG2) << "new node generated with id: "
				<< tempNodeId;
	}

	nodeIdsType_[tempNodeId] = nodeType;

	//increment the max NodeId to make sure that it is not assigned twice.
	incrementMaxNodeId();
	// wait for all threads to finish
	utilities::MPIProxy().barrier();
	return tempNodeId;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::setNodeExternalPrecursor(
	NodeId node, const WeightValue& weight) {
	if (_nodeDistribution.isLocalNode(node)) {
		if (_localNodes.count(node) > 0) {
			_localNodes.find(node)->second.setExternalPrecursor(weight, NEUTRAL);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << node
					<< "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}
	_externalReceivers.push_back(node);
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::setNodeExternalSuccessor(NodeId node) {
	_externalSenders.push_back(node);
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::makeFirstInputOfSecond(
		NodeId first, NodeId second, const WeightValue& weight) {

	//Make sure that the node exists and then add the successor
	if (_nodeDistribution.isLocalNode(first)) {
		if (_localNodes.count(first) > 0) {
			_localNodes.find(first)->second.addSuccessor(second);
			LOG(utilities::logDEBUG2)
					<< "make first input of second called first: " << first
					<< "; second: " << second;
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << first << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}

	// Make sure the Dales Law holds
	if (_nodeDistribution.isLocalNode(first)) {
		if (isDalesLawSet()) {

			auto tempNode = _localNodes.find(first)->second;

			if ((IsExcitatory(tempNode.getNodeType())  && toEfficacy(weight) < 0)
					|| (IsInhibitory(tempNode.getNodeType())
							&& toEfficacy(weight) > 0)) {
				throw utilities::Exception("Dale's law violated");

			}
		}

	}

	// Make sure that the second node exist and then set the precursor
	if (_nodeDistribution.isLocalNode(second)) {
		if (_localNodes.count(second) > 0) {
			_localNodes.find(second)->second.addPrecursor(first, weight,
					nodeIdsType_[second]);
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

	_i_report     = 0;
	_i_state      = 0;
	_i_simulation = 0;

	_parameterSimulationRun = simParam;

	_n_sim_steps            = static_cast<unsigned long>(simParam.getTEnd()/simParam.getTStep());
	_n_report_steps         = static_cast<MPILib::Number>(simParam.getTReport()/simParam.getTStep());
	_n_state_steps          = static_cast<MPILib::Number>(simParam.getTState()/simParam.getTStep());

	this->initializeLogStream(simParam.getLogName());

	try {
		//loop over all local nodes!
		for (auto& it : _localNodes) {
			it.second.configureSimulationRun(simParam);
		}

	} catch (...) {
		LOG(utilities::logERROR) << "error during configuration";
	}
	_stateNetwork.toggleConfigured();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::getExternalActivities(){

	if (_nodeDistribution.isMaster()) {
		int i=0;
		for (auto& id : _externalSenders) {
			if (_nodeDistribution.isLocalNode(id)) {
				_current_activities[i] = _localNodes.find(id)->second.getActivity();
			} else {
				utilities::MPIProxy().irecv(_nodeDistribution.getResponsibleProcessor(id), 999,
						_current_activities[i]);
			}
			i++;
		}
	} else {
		for (auto& id : _externalSenders) {
			if (_nodeDistribution.isLocalNode(id)) {
				utilities::MPIProxy().isend(0, 999, _localNodes.find(id)->second.getActivity());
			}
		}
	}
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::setExternalPrecursorActivities(
std::vector<ActivityType> activities) {

	// Check that sizes match up
	if (_externalReceivers.size() != activities.size()) {
		std::string error_string = std::string("Number of external inputs (") + std::to_string(activities.size()) 
			+ std::string(") does not match the number of nodes expected to receive them (") + std::to_string(_externalReceivers.size()) + std::string(").");
		throw std::runtime_error(error_string);
	}

	if (_nodeDistribution.isMaster()) {
		int i=0;
		for (auto& id : _externalReceivers) {
			if (_nodeDistribution.isLocalNode(id)) {
				_localNodes.find(id)->second.setExternalPrecurserActivity(activities[i]);
			} else {
				utilities::MPIProxy().isend(_nodeDistribution.getResponsibleProcessor(id), 999,
						activities[i]);
			}
			i++;
		}
	} else {
		for (auto& id : _externalReceivers) {
			if (_nodeDistribution.isLocalNode(id)) {
				_localNodes.find(id)->second.recvExternalPrecurserActivity(0,999);
			}
		}
	}
}

template<class WeightValue, class NodeDistribution>
long MPINetwork<WeightValue, NodeDistribution>::startSimulation() {
	if (_stateNetwork.isConfigured()) {
		_stateNetwork.toggleConfigured();
		LOG(utilities::logINFO) << "Starting simulation";
		// the report time must be taken as a hint if the network time step is larger (MdK: 31/08/2017)
		return static_cast<long>(_parameterSimulationRun.getTEnd()/_parameterSimulationRun.getTStep());
	}
	return 0;
}

template<class WeightValue, class NodeDistribution>
std::vector<ActivityType> MPINetwork<WeightValue, NodeDistribution>::evolveSingleStep(std::vector<ActivityType> activity) {
	try {
		LOG(utilities::logDEBUG)
				<< "****** one evolve step finished ******";

		// business as usual: keep evolving, as long as there is nothing to report
		// or to update
		updateSimulationTime();

		// Send activity to each node from process 0 from external systems
		// such as TVB
		setExternalPrecursorActivities(activity);

		MPINode<WeightValue, NodeDistribution>::waitAll();

		for (auto& it : _localNodes)
			it.second.prepareEvolve();

		Time t_current = getCurrentSimulationTime()*_parameterSimulationRun.getTStep();
		//evolve all local nodes
		for (auto& it : _localNodes)
			it.second.evolve(t_current);

		// now there is something to report or to update
		if (getCurrentSimulationTime() >= getCurrentReportTime()) {
			// there is something to report
			// CheckPercentageAndLog(CurrentSimulationTime());
			collectReport(report::RATE);
			updateReportTime();
		}

		// just a rate or also a state?
		if (getCurrentSimulationTime() >= getCurrentStateTime()) {
			// a rate as well as a state
			collectReport(report::STATE);
			updateStateTime();
		}

		collectReport(report::STATE);

		// Send activity of each node to process 0 to be passed to external systems
		// such as TVB
		_current_activities = std::vector<ActivityType>(_externalSenders.size());

		getExternalActivities();

		utilities::MPIProxy().waitAll();

		return _current_activities;
	}
	catch (utilities::IterationNumberException &e) {
		LOG(utilities::logWARNING) << "NUMBER OF ITERATIONS EXCEEDED\n";
		_stateNetwork.setResult(NUMBER_ITERATIONS_ERROR);
		return (std::vector<ActivityType>());
	}

}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::endSimulation() {
	clearSimulation();
	LOG(utilities::logINFO) << "Simulation ended, no problems noticed";
	LOG(utilities::logINFO) << "End time: "
			<< getCurrentSimulationTime() << "\n";
}

//! Envolve the network
template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::evolve() {

	if (_stateNetwork.isConfigured()) {
		_stateNetwork.toggleConfigured();

		LOG(utilities::logINFO) << "Starting simulation";
		try {
			// the report time must be taken as a hint if the network time step is larger (MdK: 31/08/2017)
			long count = (_parameterSimulationRun.getTReport() < _parameterSimulationRun.getTStep()) ? \
					      static_cast<long>(_parameterSimulationRun.getTEnd()/_parameterSimulationRun.getTStep()) : \
					      static_cast<long>(_parameterSimulationRun.getTEnd()/_parameterSimulationRun.getTReport());

			utilities::ProgressBar pb(count);
			do {
				do {

					LOG(utilities::logDEBUG)
							<< "****** one evolve step finished ******";

					// business as usual: keep evolving, as long as there is nothing to report
					// or to update
					updateSimulationTime();

					MPINode<WeightValue, NodeDistribution>::waitAll();

					for (auto& it : _localNodes)
						it.second.prepareEvolve();

					Time t_current = getCurrentSimulationTime()*_parameterSimulationRun.getTStep();
					//evolve all local nodes
					for (auto& it : _localNodes)
						it.second.evolve(t_current);

 				} while (getCurrentSimulationTime() < getCurrentReportTime()
						&& getCurrentSimulationTime() < getCurrentStateTime());

				// now there is something to report or to update
				if (getCurrentSimulationTime() >= getCurrentReportTime()) {
					// there is something to report
					// CheckPercentageAndLog(CurrentSimulationTime());
					collectReport(report::RATE);
					updateReportTime();
					pb++;
				}

				// just a rate or also a state?
				if (getCurrentSimulationTime() >= getCurrentStateTime()) {
					// a rate as well as a state
					collectReport(report::STATE);
					updateStateTime();
				}


			} while (getCurrentSimulationTime() <= getEndTime()); // it is better to test on simulation time (22/04/2017) MdK
			// write out the final state
			collectReport(report::STATE);
		}

		catch (utilities::IterationNumberException &e) {
			LOG(utilities::logWARNING) << "NUMBER OF ITERATIONS EXCEEDED\n";
			_stateNetwork.setResult(NUMBER_ITERATIONS_ERROR);
		}

		clearSimulation();
		LOG(utilities::logINFO) << "Simulation ended, no problems noticed";
		LOG(utilities::logINFO) << "End time: "
				<< getCurrentSimulationTime() << "\n";
	}

}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::reportNodeActivities(std::vector<MPILib::NodeId>& node_ids, std::vector<double>& intervals, double time) {
	if(_node_rate_current_time.size() == 0)
		_node_rate_current_time = std::vector<MPILib::Time>(node_ids.size());

	for (int i=0; i<node_ids.size(); i++){
		if (!_localNodes.count(node_ids[i]))
			continue;

		if(time < _node_rate_current_time[i]+intervals[i])
			continue;

		_node_rate_current_time[i] += intervals[i];

		std::ostringstream ost2;
		ost2 << "rate_" << _localNodes.at(node_ids[i]).getNodeId();
		std::ofstream ofst_rate(ost2.str(), std::ofstream::app);
		ofst_rate.precision(10);
		ofst_rate << _node_rate_current_time[i] << "\t" << _localNodes.at(node_ids[i]).getActivity() << std::endl;
		ofst_rate.close();
	}
}

template<class WeightValue, class NodeDistribution>
int MPINetwork<WeightValue, NodeDistribution>::getMaxNodeId() {
	utilities::MPIProxy().broadcast(_maxNodeId, 0);
	return _maxNodeId;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::incrementMaxNodeId() {
	if (_nodeDistribution.isMaster()) {
		_maxNodeId++;
	}
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::collectReport(
		report::ReportType type) {

	for (auto& it : _localNodes) {
		it.second.reportAll(type);
	}
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::initializeLogStream(
		const std::string & filename) {
	// resource will be passed on to _stream_log
	//only if filename is provided
	if (!filename.empty()) {
		std::shared_ptr<std::ostream> p_stream(
				new std::ofstream(filename.c_str()));
		if (!p_stream)
			throw utilities::Exception("MPINetwork cannot open log file.");
		utilities::Log::setStream(p_stream);
	}
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::clearSimulation() {

	for (auto& it : _localNodes) {
		it.second.clearSimulation();
	}

}

template<class WeightValue, class NodeDistribution>
bool MPINetwork<WeightValue, NodeDistribution>::isDalesLawSet() const {
	return _isDalesLaw;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::setDalesLaw(bool b_law) {
	_isDalesLaw = b_law;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateReportTime() {
	_i_report += _n_report_steps;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateSimulationTime() {
	++_i_simulation;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateStateTime() {
	_i_state += _n_state_steps;
}

template<class WeightValue, class NodeDistribution>
Index MPINetwork<WeightValue, NodeDistribution>::getEndTime() const {
	return _n_sim_steps;
}

template<class WeightValue, class NodeDistribution>
Index MPINetwork<WeightValue, NodeDistribution>::getCurrentReportTime() const {
	return _i_report;
}

template<class WeightValue, class NodeDistribution>
Index MPINetwork<WeightValue, NodeDistribution>::getCurrentSimulationTime() const {
	return _i_simulation;
}

template<class WeightValue, class NodeDistribution>
Index MPINetwork<WeightValue, NodeDistribution>::getCurrentStateTime() const {
	return _i_state;
}

template<class WeightValue, class NodeDistribution>
std::map<NodeId, MPINode<WeightValue, NodeDistribution>> MPINetwork<WeightValue,NodeDistribution>::_localNodes;

template<class WeightValue, class NodeDistribution>
NodeDistribution MPINetwork<WeightValue, NodeDistribution>::_nodeDistribution;
}

#endif //MPILIB_MPINETWORK_CODE_HPP_
