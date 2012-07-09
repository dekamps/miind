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
#include <memory>
#include <boost/timer/timer.hpp>
#include <boost/mpi/request.hpp>

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/NodeType.hpp>

#include <MPILib/include/BasicTypes.hpp>

namespace MPILib {

/**
 * MPINode the class for the actual network nodes which are distributed with mpi.
 */
template<class Weight, class NodeDistribution>
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
	explicit MPINode(const algorithm::AlgorithmInterface<Weight>& algorithm,
			NodeType nodeType, NodeId nodeId,
			const std::shared_ptr<NodeDistribution>& nodeDistribution,
			const std::shared_ptr<
					std::map<NodeId, MPINode<Weight, NodeDistribution>>>& localNode);

			/**
			 * Destructor
			 */
			virtual ~MPINode();

			/**
			 * Evolve this algorithm over a time
			 * @param time Time until the algorithm should evolve
			 * @return Time the algorithm have evolved
			 */
			Time evolve(Time time);

			/**
			 * Called before each evolve call during each evolve. Can
			 * be used to prepare the input for the evolve method.
			 */
			void prepareEvolve();

			/**
			 * Configure the Node with the Simulation Parameters
			 * @param simParam Simulation Parameters
			 */
			void configureSimulationRun(const SimulationRunParameter& simParam);

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

			/**
			 * Report the node state
			 * @param type The type of Report
			 * @return The report
			 */
			std::string reportAll (report::ReportType type) const;

			/**
			 * finishes the simulation.
			 */
			void clearSimulation();

			/**
			 * returns the type of the node
			 */
			NodeType getNodeType() const;

		private:

			/**
			 * Wait that all communication is finished
			 */
			void waitAll();

			/**
			 * exchange the Node Types
			 *
			 * It send the own NodeType to remote nodes and collects the NodeTypes of
			 * the precursor nodes and store them in the vector _precursorTypes
			 */
			void exchangeNodeTypes();

			/**
			 * Store the nodeIds of the Precursors
			 */
			std::vector<NodeId> _precursors;

			/**
			 * Store the weights of the connections to the precursors
			 */
			std::vector<Weight> _weights;

			/**
			 * Store the _precursorTypes
			 */
			std::vector<NodeType> _precursorTypes;

			/**
			 * Store the nodeIds of the successors
			 */
			std::vector<NodeId> _successors;

			// Timers for mpi and algorithm time
			static boost::timer::cpu_timer _mpiTimer, _algorithmTimer;

			// make sure that the log is only printed ones.
			static bool _isLogPrinted;

			/**
			 * A Pointer that holds the Algorithm
			 */
			boost::shared_ptr<algorithm::AlgorithmInterface<Weight>> _pAlgorithm;

			/**
			 * The type of this node needed for dales law
			 */
			NodeType _nodeType;

			/**
			 * the Id of this node
			 */
			NodeId _nodeId;

			/**
			 * Pointer to the local nodes of the processor. They are stored by the network.
			 */
			std::shared_ptr<std::map<NodeId, MPINode>> _pLocalNodes;

			//this need to be a shared_ptr see here why auto_ptr does not work:
			//http://stackoverflow.com/a/10894173/992460
			std::shared_ptr<NodeDistribution> _pNodeDistribution;

			/**
			 * Activity of this node
			 */
			ActivityType _activity;

			/**
			 * Storage for the state of the precursors, to avoid to much communication.
			 */
			std::vector<ActivityType> _precursorActivity;

			/**
			 * Store the mpi request to have assynchronous communication
			 */
			static std::vector<boost::mpi::request> _mpiStatus;

			Number _number_iterations;
			Number _maximum_iterations;

			/**
			 * True if the node Types are exchanged
			 */
			bool _isInitialised = false;

			/**
			 * Pointer to the Report Handler
			 */
			boost::shared_ptr<report::handler::AbstractReportHandler> _pHandler;
		};

typedef MPINode<double, utilities::CircularDistribution> D_MPINode;

} //end namespace

#endif /* MPILIB_MPINODE_HPP_ */
