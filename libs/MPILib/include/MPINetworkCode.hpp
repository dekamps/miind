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

#include <MPILib/config.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/utilities/IterationNumberException.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/MPINodeCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>


namespace MPILib {

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::MPINetwork() :
		_pLocalNodes(
				new std::map<NodeId, MPINode<WeightValue, NodeDistribution>>) {
}

template<class WeightValue, class NodeDistribution>
MPINetwork<WeightValue, NodeDistribution>::~MPINetwork() {

}

template<class WeightValue, class NodeDistribution>
int MPINetwork<WeightValue, NodeDistribution>::addNode(
		const algorithm::AlgorithmInterface<WeightValue>& alg,
		NodeType nodeType) {

	assert(
			nodeType == EXCITATORY || nodeType == INHIBITORY || nodeType == NEUTRAL || nodeType == EXCITATORY_BURST || nodeType == INHIBITORY_BURST);

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

	//Make sure that the node exists and then add the successor
	if (_pNodeDistribution->isLocalNode(first)) {
		if (_pLocalNodes->count(first) > 0) {
			_pLocalNodes->find(first)->second.addSuccessor(second);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << first << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}

	// Make sure the Dales Law holds
	if (_pNodeDistribution->isLocalNode(first)) {
		if (isDalesLawSet()) {

			auto tempNode = _pLocalNodes->find(first)->second;

			if ((tempNode.getNodeType() == EXCITATORY && toEfficacy(weight) < 0)
					|| (tempNode.getNodeType() == INHIBITORY
							&& toEfficacy(weight) > 0)) {
				throw utilities::Exception("Dale's law violated");

			}
		}

	}

	// Make sure that the second node exist and then set the precursor
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
	_currentReportTime = simParam.getTReport();
	_currentSimulationTime = simParam.getTBegin();
	_parameterSimulationRun = simParam;

	initializeLogStream(simParam.getLogName());

	try {
		//loop over all local nodes!
		for (auto& it : (*_pLocalNodes)) {
			it.second.configureSimulationRun(simParam);
		}

	} catch (...) {
		_streamLog << "error during configuration/n";
		_streamLog.flush();
	}

	_stateNetwork.toggleConfigured();

	_streamLog.flush();

}

//! Envolve the network
template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::evolve() {

	std::string report;
	if (_stateNetwork.isConfigured()) {
		_stateNetwork.toggleConfigured();
		_streamLog << "Starting simulation\n";
		_streamLog.flush();
		//init the nodes
		for (auto& it : (*_pLocalNodes)) {
			it.second.initNode();
		}

		try {
			utilities::ProgressBar pb(
					getEndTime() / _parameterSimulationRun.getTReport()
							+ getEndTime()
									/ _parameterSimulationRun.getTState());

			do {
				do {
					// business as usual: keep evolving, as long as there is nothing to report
					// or to update
					updateSimulationTime();

					MPINode<WeightValue, NodeDistribution>::waitAll();

					for (auto& it : (*_pLocalNodes)) {
						it.second.prepareEvolve();
					}


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
					report = collectReport(report::RATE);
					_streamLog << report;
				}
				// just a rate or also a state?
				if (getCurrentSimulationTime() >= getCurrentStateTime()) {
					// a rate as well as a state
					collectReport(report::STATE);
					updateStateTime();
				}
				pb++;

			} while (getCurrentReportTime() < getEndTime());
			// write out the final state
			collectReport(report::STATE);
		}

		catch (utilities::IterationNumberException &e) {
			_streamLog << "NUMBER OF ITERATIONS EXCEEDED\n";
			_stateNetwork.setResult(NUMBER_ITERATIONS_ERROR);
			_streamLog.flush();
			_streamLog.close();
		}

		clearSimulation();
		_streamLog << "Simulation ended, no problems noticed\n";
		_streamLog << "End time: " << getCurrentSimulationTime() << "\n";
		_streamLog.close();
	}

}

template<class WeightValue, class NodeDistribution>
int MPINetwork<WeightValue, NodeDistribution>::getMaxNodeId() {
	utilities::MPIProxy mpiProxy;
	mpiProxy.broadcast(_maxNodeId, 0);
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
		report::ReportType type) {

	std::string string_return;

	for (auto& it : (*_pLocalNodes)) {
		string_return += it.second.reportAll(type);
	}

	return string_return;
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::initializeLogStream(
		const std::string & filename) {
	// resource will be passed on to _stream_log
	std::shared_ptr<std::ostream> p_stream(new std::ofstream(filename.c_str()));
	if (!p_stream)
		throw utilities::Exception("MPINetwork cannot open log file.");
	if (!_streamLog.openStream(p_stream))
		_streamLog << "WARNING YOU ARE TRYING TO REOPEN THIS LOG FILE\n";
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::clearSimulation() {

	for (auto& it : (*_pLocalNodes)) {
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
	_currentReportTime += _parameterSimulationRun.getTReport();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateSimulationTime() {
	_currentSimulationTime += _parameterSimulationRun.getTStep();
}

template<class WeightValue, class NodeDistribution>
void MPINetwork<WeightValue, NodeDistribution>::updateStateTime() {
	_currentStateTime += _parameterSimulationRun.getTState();
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getEndTime() const {
	return _parameterSimulationRun.getTEnd();
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getCurrentReportTime() const {
	return _currentReportTime;
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getCurrentSimulationTime() const {
	return _currentSimulationTime;
}

template<class WeightValue, class NodeDistribution>
Time MPINetwork<WeightValue, NodeDistribution>::getCurrentStateTime() const {
	return _currentStateTime;

}

}					//end namespace MPILib

#endif //MPILIB_MPINETWORK_CODE_HPP_

