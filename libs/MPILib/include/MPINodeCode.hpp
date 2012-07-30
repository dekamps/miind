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

#ifndef CODE_MPILIB_MPINODE_HPP_
#define CODE_MPILIB_MPINODE_HPP_

#include <MPILib/config.hpp>
#include <MPILib/include/MPINode.hpp>
#include <iostream>
#include <MPILib/include/utilities/Exception.hpp>
namespace MPILib {

template<class Weight, class NodeDistribution>
MPINode<Weight, NodeDistribution>::MPINode(
		const algorithm::AlgorithmInterface<Weight>& algorithm,
		NodeType nodeType, NodeId nodeId,
		const NodeDistribution& nodeDistribution,
		const std::map<NodeId, MPINode<Weight, NodeDistribution>>& localNode) :
		_pAlgorithm(algorithm.clone()), //
		_nodeType(nodeType), //
		_nodeId(nodeId), //
		_rLocalNodes(localNode), //
		_rNodeDistribution(nodeDistribution) {
}

template<class Weight, class NodeDistribution>
MPINode<Weight, NodeDistribution>::~MPINode() {
}

template<class Weight, class NodeDistribution>
Time MPINode<Weight, NodeDistribution>::evolve(Time time) {

	while (_pAlgorithm->getCurrentTime() < time) {
		++_number_iterations;

		_pAlgorithm->evolveNodeState(_precursorActivity, _weights, time,
				_precursorTypes);

	}

	// update state
	this->setActivity(_pAlgorithm->getCurrentRate());

	sendOwnActivity();
	receiveData();

	return _pAlgorithm->getCurrentTime();
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::prepareEvolve() {

	_pAlgorithm->prepareEvolve(_precursorActivity, _weights, _precursorTypes);

}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::configureSimulationRun(
		const SimulationRunParameter& simParam) {
	_maximum_iterations = simParam.getMaximumNumberIterations();
	_pAlgorithm->configure(simParam);

	// Add this line or other nodes will not get a proper input at the first simulation step!
	this->setActivity(_pAlgorithm->getCurrentRate());

	_pHandler = std::shared_ptr<report::handler::AbstractReportHandler>(
			simParam.getHandler().clone());

	_pHandler->initializeHandler(_nodeId);

}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::addPrecursor(NodeId nodeId,
		const Weight& weight) {
	_precursors.push_back(nodeId);
	_weights.push_back(weight);
	//make sure that _precursorStates is big enough to store the data
	_precursorActivity.resize(_precursors.size());
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::addSuccessor(NodeId nodeId) {
	_successors.push_back(nodeId);
}

template<class Weight, class NodeDistribution>
ActivityType MPINode<Weight, NodeDistribution>::getActivity() const {
	return _activity;
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::setActivity(ActivityType activity) {
	_activity = activity;
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::waitAll() {
	utilities::MPIProxy mpiProxy;
	mpiProxy.waitAll();
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::initNode() {
	if (!_isInitialised) {
		this->exchangeNodeTypes();
		_isInitialised = true;
#ifdef DEBUG
		std::cout << "init finished. Node Types lenght: "
		<< _precursorTypes.size() << " number of precursors: "
		<< _precursors.size() << std::endl;
#endif
	} else {
		throw utilities::Exception("init called more than once.");
	}
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::exchangeNodeTypes() {
	_precursorTypes.resize(_precursors.size());
	//get the Types from the precursors
	int i = 0;
	for (auto it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		//do not send the data if the node is local!
		if (_rNodeDistribution.isLocalNode(*it)) {
			_precursorTypes[i] =
					_rLocalNodes.find(*it)->second.getNodeType();

		} else {
			utilities::MPIProxy mpiProxy;
			mpiProxy.irecv(_rNodeDistribution.getResponsibleProcessor(*it), *it,
					_precursorTypes[i]);
		}
	}
	for (auto& it : _successors) {
		//do not send the data if the node is local!
		if (!_rNodeDistribution.isLocalNode(it)) {
			utilities::MPIProxy mpiProxy;
			mpiProxy.isend(_rNodeDistribution.getResponsibleProcessor(it),
					_nodeId, _nodeType);
		}
	}
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::receiveData() {
	int i = 0;

	for (auto it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		//do not send the data if the node is local!
		if (_rNodeDistribution.isLocalNode(*it)) {
			_precursorActivity[i] =
					_rLocalNodes.find(*it)->second.getActivity();

		} else {
			utilities::MPIProxy mpiProxy;
			mpiProxy.irecv(_rNodeDistribution.getResponsibleProcessor(*it), *it,
					_precursorActivity[i]);
		}
	}
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::sendOwnActivity() {

	utilities::MPIProxy mpiProxy;
	for (auto& it : _successors) {
		//do not send the data if the node is local!
		if (!_rNodeDistribution.isLocalNode(it)) {
			mpiProxy.isend(_rNodeDistribution.getResponsibleProcessor(it),
					_nodeId, _activity);
		}
	}
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::reportAll(
		report::ReportType type) const {

	std::vector<report::ReportValue> vec_values;

	if (type == report::RATE || type == report::STATE) {
		report::Report report(_pAlgorithm->getCurrentTime(),
				Rate(this->getActivity()), this->_nodeId,
				_pAlgorithm->getGrid(),  type, vec_values,
				_rLocalNodes.size());

		_pHandler->writeReport(report);
	}
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::clearSimulation() {
	_pHandler->detachHandler(_nodeId);

}

template<class Weight, class NodeDistribution>
NodeType MPINode<Weight, NodeDistribution>::getNodeType() const {
	return _nodeType;
}

}
//end namespace MPILib

#endif /* CODE_MPILIB_MPINODE_HPP_ */
