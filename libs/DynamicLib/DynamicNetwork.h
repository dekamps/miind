// Copyright (c) 2005 - 2010 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_LIBS_DYNAMICLIB_DYNAMICNETWORK_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DYNAMICNETWORK_INCLUDE_GUARD

#include <stack>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include "../UtilLib/UtilLib.h"
#include "../NetLib/NetLib.h"
#include "AbstractAlgorithm.h"
#include "DynamicNetworkImplementation.h"
#include "DynamicNodeCode.h"
#include "InactiveReportHandler.h"
#include "NetworkState.h"
#include "ReportManager.h"
#include "SimulationRunParameter.h"

using boost::shared_ptr;
using NetLib::NodeId;
using UtilLib::LogStream;
using UtilLib::Number;
using UtilLib::Streamable;
using std::ostream;
using std::ofstream;
using std::endl;
using std::stack;

namespace DynamicLib
{
	//! With a DynamicNetwork one can simulate network processes in sparse random networks.

	//! In principle every network process can be simulated. One creates a network simulation by first
	//! configuring a DynamicNetwork with a version an AbstractAlgorithm and then one adds the DynamicNode
	//! to the DynamicNetwork. With the DynamicNetwork one can create links between nodes that already have
	//! been added to the network. The DynamicNetwork can then be configured with a SimulationRunParameter,
	//! to specify begin and end time of the simulation, the place where simulation results should be stored,
	//! etc. The network can then Evolve: the simulation is run automatically by the network. Many AbstractAlgorithms
	//! are provided with MIIND: so-called Wilson-Cowan dynamics and population density techniques, for example.
	//! This means that the user's task is mainly restricted to determine the network's architecture, but the simulation
	//! is done automatically. From the user's perspective there is no need to call numerical software directly.
	//! It is also possible to develop new AbstractAlgorithms. The user who does this only needs to develop the 
	//! new version of an AbstractAlgorithm on a single node. The network facilities of DynamicNetwork then deliver
	//! methods for simulating a network of such nodes and store the simulation results. DynamicNetwork is a framework
	//! that takes away many low level tasks, such as running the simulation loop, storing simulation results, maintaining
	//! log files, etc away from the developer. Using a RootReportHandler gives immediate acccess to visualization
	//! capabilities, including the online monitoring of simulation results. DynamicNetwork is an enveloppe class:
	//! the real work is done in DynamicNetworkImplementation. This allows for the development of multiple network 
	//! implementation ideas, whilst minimising the impact of changing a network implementation on client code.

	template <class Implementation>
	class DynamicNetwork : public Streamable
	{
	public:

	
		typedef typename Implementation::WeightType_ WeightType;
		typedef typename AbstractSparseNode<double,WeightType>::predecessor_iterator predecessor_iterator;

		//! create a DynamicNetwork with zero nodes
		DynamicNetwork
		(
		);

		//! copy constructor a dynamic network
		DynamicNetwork
		(
			const DynamicNetwork<Implementation>&
		);

		~DynamicNetwork();

		//! Add an extra node to the network
		NodeId AddNode
		(
			const AbstractAlgorithm<WeightType>&,
			NodeType
		);
		
		//! the node, labeled by the first NodeId, will be input of the node, labebled by the second NodeId
		bool MakeFirstInputOfSecond
		(
			NodeId,
			NodeId,
			const WeightType&
		);

		//! Associate a node with a SpatialPosition
		bool AssociateNodePosition
		(
			NodeId,
			const SpatialPosition&
		);

		//! If a position is associated with the NodeId, the function returns true
		//! and the associated positions is in the SpatialPosition, otherwise the function returns false and the
		//! SpatialPosition is undefined.

		bool GetPosition
		( 
			NodeId,
			SpatialPosition*
		);

		//! Set a node's name
		void SetNodeName
		(
			NodeId,
			const string&
		);
		
		//! Get a node's name
		string GetNodeName
		(
			NodeId
		);

		//! Configure the next simulation
		bool ConfigureSimulation
		(
			const SimulationRunParameter&
		);

		//!
		bool Evolve();

		//! return if nodes can only purely excitatory or purely inhibitory
		bool IsDalesLawSet() const;

		//!
		bool SetDalesLaw(bool);

		//! NodeIterator to the first node in the network
		NodeIterator<DynamicNode<WeightType> > begin();

		//! NodeIterator one past the last node in the network
		NodeIterator<DynamicNode<WeightType> > end();

		//!
		predecessor_iterator begin(NodeId);

		//!
		predecessor_iterator end(NodeId);

		//!
		NodeState State(NodeId) const;

		//!
		virtual bool ToStream (ostream&) const;

		//!
		virtual bool FromStream(istream&);

		//!
		virtual string Tag() const;

		//! Number of nodes in network
		Number NumberOfNodes() const;

	private:

		typedef pair<float, Time> TimePercentPair;
		
		DynamicNetwork<Implementation>& 
			operator=
			(
				const DynamicNetwork<Implementation>&
			);

		Time EndTime						() const;
		Time CurrentReportTime				() const;
		Time CurrentSimulationTime			() const;
		Time CurrentUpdateTime				() const;
		Time CurrentStateTime				() const;

		void UpdateReportTime				();
		void UpdateSimulationTime			();
		void UpdateUpdateTime				();
		void UpdateStateTime				();

		void InitializePercentageQueue		();
		void InitializeLogStream			(const string&);
		void HandleIterationNumberException ();
		void CheckPercentageAndLog			(Time time);

		Time  PercentageTimeTop				() const;
		float CurrentPercentage				() const;
		void  AdaptPercentage				();
		
		Time					_current_report_time;
		Time					_current_update_time;
		Time					_current_state_time;
		Time					_current_simulation_time;


		SimulationRunParameter	_parameter_simulation_run;
		NetworkState			_state_network;
		Implementation			_implementation;
		LogStream				_stream_log;

		stack<TimePercentPair>	_stack_percentage;

	}; // end of DynamicNetwork

	template <class WeightValue>
	ostream& operator<<(ostream& s, DynamicNetwork<WeightValue>& net)
	{
		net.ToStream(s);
		return s;
	}

	typedef DynamicNetwork<DynamicNetworkImplementation<double> > D_DynamicNetwork;

} // end of DynamicLib

#endif // include guard
