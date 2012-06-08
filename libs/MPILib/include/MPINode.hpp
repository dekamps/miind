/*
 * MPINode.h
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#ifndef MPILIB_MPINODE_HPP_
#define MPILIB_MPINODE_HPP_

#include <vector>
#include <map>
#include <boost/mpi/request.hpp>
#include <boost/shared_ptr.hpp>

#include <MPILib/include/AlgorithmInterface.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>

#include <MPILib/include/BasicTypes.hpp>

#include <DynamicLib/NodeState.h>

namespace MPILib {

/**
 * @class MPINode the class for the actual network nodes. T
 */
template <class Weight, class NodeDistribution>
class MPINode {
public:
	/**
	 * Constructor
	 * @param algorithm Algorithm the algorithm the node should contain
	 * @param nodeType NodeType the type of the node
	 * @param nodeId NodeId the id of the node
	 * @param nodeDistribution The Node Distribution.
	 * @param localNode The local nodes of this processor
	 */
	explicit MPINode(const AlgorithmInterface<Weight>& algorithm, NodeType nodeType,
			NodeId nodeId,
			const boost::shared_ptr<NodeDistribution>& nodeDistribution,
			const boost::shared_ptr<std::map<NodeId, MPINode<Weight, NodeDistribution> > >& localNode);


	/**
	 * Destructor
	 */
	virtual ~MPINode();

	/**
	 * Evolve this algorithm over a time
	 * @param time Time until the algorithm should evolve
	 * @return Time the algorithm have evolved
	 */
	Time Evolve(Time time);

	/**
	 * Configure the Node with the Simulation Parameters
	 * @param simParam Simulation Parameters
	 */
	void ConfigureSimulationRun(const DynamicLib::SimulationRunParameter& simParam);

	/**
	 * Add a precursor to the current node
	 * @param nodeId NodeId the id of the precursor
	 * @param weight the weight of the connection
	 */
	void addPrecursor(NodeId nodeId, const Weight& weight);

	/**
	 * Add a successor to the current node
	 * @param nodeId NodeId the id of the successor
	 */
	void addSuccessor(NodeId nodeId);

//	/**
//	 * Getter for the Nodes state
//	 * @return The current node state
//	 */
//	DynamicLib::NodeState getState() const;
//
//	/**
//	 * The Setter for the node state
//	 * @param state The state the node should be in
//	 */
//	void setState(DynamicLib::NodeState state);
	/**
	 * Getter for the Nodes activity
	 * @return The current node activity
	 */
	ActivityType getActivity() const;

	/**
	 * The Setter for the node activity
	 * @param activity The activity the node should be in
	 */
	void setActivity(ActivityType activity);


	/**
	 * Receive the new data from the precursor nodes
	 */
	void receiveData();

	/**
	 * Send the own state to the successors.
	 */
	void sendOwnActivity();

	std::string reportAll	(DynamicLib::ReportType) const;

	void clearSimulation();


private:

	void waitAll();

	std::vector<NodeId> _precursors;

	std::vector<Weight> _weights;

	std::vector<NodeId> _successors;


	NodeType _nodeType;

	/**
	 * the Id of this node
	 */
	NodeId _nodeId;

	/**
	 * Pointer to the local nodes of the processor. They are stored by the network.
	 */
	boost::shared_ptr<std::map<NodeId, MPINode> > _pLocalNodes;

	//this need to be a shared_ptr see here why auto_ptr does not work:
	//http://stackoverflow.com/a/10894173/992460
	boost::shared_ptr<NodeDistribution> _pNodeDistribution;



	/**
	 * Activity of this node
	 */
	ActivityType _activity;

	/**
	 * Storage for the state of the precursors, to avoid to much communication.
	 */
	std::vector<ActivityType> _precursorActivity;

	std::vector<boost::mpi::request> _mpiStatus;

	UtilLib::Number						_number_iterations;
	UtilLib::Number						_maximum_iterations;
	NodeType							_type;
	DynamicLib::NodeInfo				_info;

	boost::shared_ptr<AlgorithmInterface<Weight> > _algorithm;
	mutable boost::shared_ptr<DynamicLib::AbstractReportHandler>	_p_handler;
};

typedef MPINode<double, utilities::CircularDistribution> D_MPINode;


} //end namespace

#endif /* MPILIB_MPINODE_HPP_ */
