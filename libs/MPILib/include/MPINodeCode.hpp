/*
 * MPINode.cpp
 *
 *  Created on: 29.05.2012
 *      Author: David Sichau
 */

#ifndef CODE_MPILIB_MPINODE_HPP_
#define CODE_MPILIB_MPINODE_HPP_

#include <MPILib/include/MPINode.hpp>
#include <iostream>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>

namespace mpi = boost::mpi;
namespace MPILib {

template<class Weight, class NodeDistribution>
MPINode<Weight, NodeDistribution>::MPINode(
		const algorithm::AlgorithmInterface<Weight>& algorithm,
		NodeType nodeType, NodeId nodeId,
		const std::shared_ptr<NodeDistribution>& nodeDistribution,
		const std::shared_ptr<
				std::map<NodeId, MPINode<Weight, NodeDistribution>>>& localNode) :
		_pAlgorithm(algorithm.clone()), //
		_nodeType(nodeType),//
		_nodeId(nodeId),//
		_pLocalNodes(localNode),//
		_pNodeDistribution(nodeDistribution)
		{

		}

template<class Weight, class NodeDistribution>
MPINode<Weight, NodeDistribution>::~MPINode() {
}

template<class Weight, class NodeDistribution>
Time MPINode<Weight, NodeDistribution>::evolve(Time time) {

	receiveData();

	waitAll();
	while (_pAlgorithm->getCurrentTime() < time) {
		++_number_iterations;

		_pAlgorithm->evolveNodeState(_precursorActivity, _weights, time);

	}

	// update state
	this->setActivity(_pAlgorithm->getCurrentRate());

	sendOwnActivity();

	return _pAlgorithm->getCurrentTime();
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::configureSimulationRun(
		const SimulationRunParameter& simParam) {
	_maximum_iterations = simParam.MaximumNumberIterations();
	_pAlgorithm->configure(simParam);

	// Add this line or other nodes will not get a proper input at the first simulation step!
	this->setActivity(_pAlgorithm->getCurrentRate());

	_pHandler = boost::shared_ptr<AbstractReportHandler>(
			simParam.Handler().clone());

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
	mpi::wait_all(_mpiStatus.begin(), _mpiStatus.end());
	_mpiStatus.clear();

}
template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::receiveData() {

	int i = 0;
	for (auto it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		mpi::communicator world;
		//do not send the data if the node is local!
		if (_pNodeDistribution->isLocalNode(*it)) {
			_precursorActivity[i] =
					_pLocalNodes->find(*it)->second.getActivity();

		} else {
			_mpiStatus.push_back(
					world.irecv(
							_pNodeDistribution->getResponsibleProcessor(*it),
							_pNodeDistribution->getRank(),
							_precursorActivity[i]));
		}
	}
}
template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::sendOwnActivity() {
	mpi::communicator world;

	for (auto& it : _successors) {
		//do not send the data if the node is local!
		if (!_pNodeDistribution->isLocalNode(it)) {
			_mpiStatus.push_back(
					world.isend(_pNodeDistribution->getResponsibleProcessor(it),
							it, _activity));
		}
	}

}

template<class Weight, class NodeDistribution>
std::string MPINode<Weight, NodeDistribution>::reportAll(
		ReportType type) const {

	std::string string_return("");

	std::vector<ReportValue> vec_values;

	if (type == RATE || type == STATE) {
		Report report(_pAlgorithm->getCurrentTime(),
				Rate(this->getActivity()),
				this->_nodeId,
				_pAlgorithm->getGrid(), string_return, type, vec_values);

		_pHandler->writeReport(report);
	}


	return string_return;
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::clearSimulation() {
	_pHandler->detachHandler(_nodeId);
}

} //end namespace MPILib

#endif /* CODE_MPILIB_MPINODE_HPP_ */
