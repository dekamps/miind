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
			//hope this sets the timer to zero.
			_mpiTimer.start();
			_mpiTimer.stop();
			_algorithmTimer.start();
			_algorithmTimer.stop();

		}

template<class Weight, class NodeDistribution>
MPINode<Weight, NodeDistribution>::~MPINode() {
}

template<class Weight, class NodeDistribution>
Time MPINode<Weight, NodeDistribution>::evolve(Time time) {

	receiveData();

	waitAll();

	_algorithmTimer.resume();

	while (_pAlgorithm->getCurrentTime() < time) {
		++_number_iterations;

		_pAlgorithm->evolveNodeState(_precursorActivity, _weights, time);

	}

	// update state
	this->setActivity(_pAlgorithm->getCurrentRate());

	_algorithmTimer.stop();


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

	_pHandler = boost::shared_ptr<report::handler::AbstractReportHandler>(
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

	_mpiTimer.resume();

	mpi::wait_all(_mpiStatus.begin(), _mpiStatus.end());
	_mpiStatus.clear();

	_mpiTimer.stop();

}
template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::receiveData() {
	_mpiTimer.resume();

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

	_mpiTimer.stop();
}
template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::sendOwnActivity() {
	_mpiTimer.resume();

	mpi::communicator world;

	for (auto& it : _successors) {
		//do not send the data if the node is local!
		if (!_pNodeDistribution->isLocalNode(it)) {
			_mpiStatus.push_back(
					world.isend(_pNodeDistribution->getResponsibleProcessor(it),
							it, _activity));
		}
	}

	_mpiTimer.stop();
}

template<class Weight, class NodeDistribution>
std::string MPINode<Weight, NodeDistribution>::reportAll(
		report::ReportType type) const {

	std::string string_return("");

	std::vector<report::ReportValue> vec_values;

	if (type == report::RATE || type == report::STATE) {
		report::Report report(_pAlgorithm->getCurrentTime(), Rate(this->getActivity()),
				this->_nodeId, _pAlgorithm->getGrid(), string_return, type,
				vec_values);

		_pHandler->writeReport(report);
	}

	return string_return;
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::clearSimulation() {
	_pHandler->detachHandler(_nodeId);

	if(_pNodeDistribution->isMaster() && !_isLogPrinted){
		std::cout<<"MPI Timer: "<<_mpiTimer.format()<<std::endl;
		std::cout<<"Algorithm Timer: "<<_algorithmTimer.format()<<std::endl;
		_isLogPrinted=true;

	}
}

template<class Weight, class NodeDistribution>
boost::timer::cpu_timer MPINode<Weight, NodeDistribution>::_mpiTimer = boost::timer::cpu_timer();

template<class Weight, class NodeDistribution>
boost::timer::cpu_timer MPINode<Weight, NodeDistribution>::_algorithmTimer = boost::timer::cpu_timer();

template<class Weight, class NodeDistribution>
bool MPINode<Weight, NodeDistribution>::_isLogPrinted = false;

} //end namespace MPILib

#endif /* CODE_MPILIB_MPINODE_HPP_ */
