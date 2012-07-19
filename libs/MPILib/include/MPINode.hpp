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

#ifndef MPILIB_MPINODE_HPP_
#define MPILIB_MPINODE_HPP_

#include <MPILib/config.hpp>
#include <vector>
#include <map>
#include <memory>
#include <boost/timer/timer.hpp>

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/NodeType.hpp>

#include <MPILib/include/TypeDefinitions.hpp>


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

			/**
			 * Wait that all communication is finished
			 */
			static void waitAll();

			void initNode();

		private:



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
			static boost::timer::cpu_timer _algorithmTimer;

			// make sure that the log is only printed ones.
			static bool _isLogPrinted;

			/**
			 * A Pointer that holds the Algorithm
			 */
			std::shared_ptr<algorithm::AlgorithmInterface<Weight>> _pAlgorithm;

			/**
			 * The type of this node needed for dales law
			 */
			NodeType _nodeType;

			/**
			 * the Id of this node
			 */
			NodeId _nodeId;

			/**
			 * Pointer to the local nodes of the processor. They are owned by the network.
			 */
			std::weak_ptr<std::map<NodeId, MPINode>> _pLocalNodes;

			/**
			 * Pointer to the NodeDistribution. This is owned by the network.
			 */
			std::weak_ptr<NodeDistribution> _pNodeDistribution;

			/**
			 * Activity of this node
			 */
			ActivityType _activity = 0;

			/**
			 * Storage for the state of the precursors, to avoid to much communication.
			 */
			std::vector<ActivityType> _precursorActivity;

			Number _number_iterations;
			Number _maximum_iterations;

			/**
			 * True if the node Types are exchanged
			 */
			bool _isInitialised = false;

			/**
			 * Pointer to the Report Handler
			 */
			std::shared_ptr<report::handler::AbstractReportHandler> _pHandler;
		};

typedef MPINode<double, utilities::CircularDistribution> D_MPINode;

} //end namespace

#endif /* MPILIB_MPINODE_HPP_ */
