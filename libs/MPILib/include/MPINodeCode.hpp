/*
 * MPINode.cpp
 *
 *  Created on: 29.05.2012
 *      Author: David Sichau
 */

#ifndef CODE_MPILIB_MPINODE_HPP_
#define CODE_MPILIB_MPINODE_HPP_

#include <MPILib/config.hpp>
#include <MPILib/include/MPINode.hpp>
#include <iostream>
#include <MPILib/include/utilities/Exception.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
namespace mpi = boost::mpi;
#endif
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

	_algorithmTimer.resume();

	while (_pAlgorithm->getCurrentTime() < time) {
		++_number_iterations;

		_pAlgorithm->evolveNodeState(_precursorActivity, _weights, time,
				_precursorTypes);

	}

	// update state
	this->setActivity(_pAlgorithm->getCurrentRate());

	_algorithmTimer.stop();

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
#ifdef ENABLE_MPI
	_mpiTimer.resume();
	mpi::wait_all(_mpiStatus.begin(), _mpiStatus.end());
	_mpiStatus.clear();

	_mpiTimer.stop();
#endif

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
#ifdef ENABLE_MPI
	mpi::communicator world;
	_precursorTypes.resize(_precursors.size());
	//get the Types from the precursors
	int i = 0;
	for (auto it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		//do not send the data if the node is local!
		if (_pNodeDistribution->isLocalNode(*it)) {
			_precursorTypes[i] = _pLocalNodes->find(*it)->second.getNodeType();

		} else {
#ifdef DEBUG
			std::cout << "resv source: "
			<< _pNodeDistribution->getResponsibleProcessor(*it) << "\ttag: "
			<< *it <<"\tnodeID: "<<world.rank()<< std::endl;
#endif
			_mpiStatus.push_back(
					world.irecv(
							_pNodeDistribution->getResponsibleProcessor(*it),
							*it, _precursorTypes[i]));
		}
	}
	//send own types to successors
	for (auto& it : _successors) {
		//do not send the data if the node is local!
		if (!_pNodeDistribution->isLocalNode(it)) {
#ifdef DEBUG
			std::cout << "send dest: "
			<< _pNodeDistribution->getResponsibleProcessor(it) << "\ttag: "
			<< _nodeId <<"\tnodeID: "<<world.rank()<< std::endl;
#endif
			_mpiStatus.push_back(
					world.isend(_pNodeDistribution->getResponsibleProcessor(it),
							_nodeId, _nodeType));
		}
	}
#else
	_precursorTypes.resize(_precursors.size());
	//get the Types from the precursors
	int i = 0;
	for (auto it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		_precursorTypes[i] = _pLocalNodes->find(*it)->second.getNodeType();
	}
#endif

}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::receiveData() {

#ifdef ENABLE_MPI

	_mpiTimer.resume();
	mpi::communicator world;

	int i = 0;
	for (auto it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		//do not send the data if the node is local!
		if (_pNodeDistribution->isLocalNode(*it)) {
			_precursorActivity[i] =
			_pLocalNodes->find(*it)->second.getActivity();

		} else {
#ifdef DEBUG
			std::cout << "resv source: "
			<< _pNodeDistribution->getResponsibleProcessor(*it) << "\ttag: "
			<< *it <<"\tnodeID: "<<world.rank()<< std::endl;
#endif
			_mpiStatus.push_back(
					world.irecv(
							_pNodeDistribution->getResponsibleProcessor(*it),
							*it,
							_precursorActivity[i]));
		}
	}
	_mpiTimer.stop();
#else
	int i = 0;
	for (auto it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		_precursorActivity[i] = _pLocalNodes->find(*it)->second.getActivity();
	}
#endif
}
template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::sendOwnActivity() {
#ifdef ENABLE_MPI

	_mpiTimer.resume();
	mpi::communicator world;

	for (auto& it : _successors) {
		//do not send the data if the node is local!
		if (!_pNodeDistribution->isLocalNode(it)) {
#ifdef DEBUG
			std::cout << "send dest: "
			<< _pNodeDistribution->getResponsibleProcessor(it) << "\ttag: "
			<< _nodeId <<"\tnodeID: "<<world.rank()<< std::endl;
#endif
			_mpiStatus.push_back(
					world.isend(_pNodeDistribution->getResponsibleProcessor(it),
							_nodeId, _activity));
		}
	}
	_mpiTimer.stop();
#endif
}

template<class Weight, class NodeDistribution>
std::string MPINode<Weight, NodeDistribution>::reportAll(
		report::ReportType type) const {

	std::string string_return("");

	std::vector<report::ReportValue> vec_values;

	if (type == report::RATE || type == report::STATE) {
		report::Report report(_pAlgorithm->getCurrentTime(),
				Rate(this->getActivity()), this->_nodeId,
				_pAlgorithm->getGrid(), string_return, type, vec_values,
				_pLocalNodes->size());

		_pHandler->writeReport(report);
	}

	return string_return;
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::clearSimulation() {
	_pHandler->detachHandler(_nodeId);

	if (_pNodeDistribution->isMaster() && !_isLogPrinted) {
		std::cout << "MPI Timer: " << _mpiTimer.format() << std::endl;
		std::cout << "Algorithm Timer: " << _algorithmTimer.format()
				<< std::endl;
		_isLogPrinted = true;

	}
}

template<class Weight, class NodeDistribution>
NodeType MPINode<Weight, NodeDistribution>::getNodeType() const {
	return _nodeType;
}

template<class Weight, class NodeDistribution>
boost::timer::cpu_timer MPINode<Weight, NodeDistribution>::_mpiTimer =
		boost::timer::cpu_timer();

template<class Weight, class NodeDistribution>
boost::timer::cpu_timer MPINode<Weight, NodeDistribution>::_algorithmTimer =
		boost::timer::cpu_timer();

template<class Weight, class NodeDistribution>
bool MPINode<Weight, NodeDistribution>::_isLogPrinted = false;
#ifdef ENABLE_MPI
template<class Weight, class NodeDistribution>
std::vector<boost::mpi::request> MPINode<Weight, NodeDistribution>::_mpiStatus;
#endif

}
 //end namespace MPILib

#endif /* CODE_MPILIB_MPINODE_HPP_ */
