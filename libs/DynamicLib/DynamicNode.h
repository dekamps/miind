// Copyright (c) 2005 - 2011 Marc de Kamps
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
#ifndef _CODE_LIBS_DYNAMICLIB_DYNAMICNODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DYNAMICNODE_INCLUDE_GUARD

#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "AbstractAlgorithm.h"
#include "NodeInfo.h"
#include "ReportType.h"

using SparseImplementationLib::AbstractSparseNode;
using UtilLib::Number;

namespace DynamicLib
{

	//! This gives a DynamicNode's type, which will be checked when Dale's law is set.
	enum NodeType {NEUTRAL, EXCITATORY, INHIBITORY, EXCITATORY_BURST, INHIBITORY_BURST};

	//! DynamicNode is a SparseNode that can run Algorithms to evolve its state.
	//!
	//! A DynamicNode has an instance of an AbstractAlgorithm. The algorithm maintains a NodeState and
	//! is able to evolve the state over a certain time step. The DynamicNode is able to evaluate the
	//! contribution from other nodes at each time step and communicates this to the algorithm. In this 
	//! way DynamicNodes can influence each other. When they are grouped together in a DynamicNetwork,
	//! the DynamicNetwork is able to drive a network simulation, by driving individual DynamicNodes.

	template <class Weight>
	class DynamicNode : public AbstractSparseNode<double,Weight>
	{
	public:

		//! default constructor is necessary for vector placement
		DynamicNode();

		//! A DynamicNode receives its own clone from an AbstractAlgorithm instance,
		//! but an indivdual set of parameters
		DynamicNode
		(
			const AbstractAlgorithm<Weight>&,	/*!< A DynamicNode must be configured with an AbstractAlgorithm						*/ 
			NodeType type						/*!< A DynamicNode is EXCITATORY or INHIBITORY unless Dale's law is switched off	*/
		);

		//! copy constructor
		DynamicNode
		(
			const DynamicNode<Weight>&
		);

		//! virtual destructor
		virtual ~DynamicNode();

		DynamicNode<Weight>& operator=(const DynamicNode<Weight>&);

		//! Evolve function, until specified some specified time,
		//! determined by DynamicNetwork dynamics. 
		Time Evolve(Time);

		//! Collect input, in case a synchrnonous network evolution is required
		bool CollectExternalInput();

		//!  Configure a Simulation
		bool ConfigureSimulationRun
		(
			const SimulationRunParameter&
		);

		//! Give a node a name
		void SetNodeName(const string&);

		//! Get a node's name
		string GetNodeName() const;

		//! Associate a spatial position with a node
		void AssociatePosition(const SpatialPosition&);

		//! Get the SpatialPosition associated with thes Node, if it exists.
		//! Function returns true if the position exists and false if not. The resulting position is then undefined
		bool GetPosition(SpatialPosition*) const;

		NodeType Type		() const;

		Time CurrentTime	() const;

		template <class StateType> 
			StateType  State	() const;

		bool ClearSimulation();

		bool UpdateHandler	();

		string ReportAll	(ReportType) const;

		//! it is sometimes necessary to retrieve the properties of an algorithm, but direct access is forbidden
		auto_ptr<AbstractAlgorithm<Weight> > CloneAlgorithm() const;

		//! get a node from a stream
		virtual bool FromStream(istream&);

		//! write a node to a stream
		virtual bool ToStream(ostream&) const;

		//! tag for serialization (see Util::Streamable)
		virtual string Tag() const;

	protected:

		virtual DynamicNode<Weight>*	Address(std::ptrdiff_t); 
		virtual std::ptrdiff_t			Offset(AbstractSparseNode<double,Weight>*) const;

	private:
		
		Number	NumberMaximumIterations() const;

		auto_ptr<AbstractAlgorithm<Weight> > AbsorbAlgorithm(istream&);

		Number						_number_iterations;
		Number						_maximum_iterations;
		NodeType					_type;
		NodeState					_state;
		NodeInfo					_info;
		string						_name;

		auto_ptr<AbstractAlgorithm<Weight> >	_p_algorithm;
		mutable auto_ptr<AbstractReportHandler>	_p_handler;
		
	};

	typedef DynamicNode<double> D_DynamicNode;

} // end of DynamicLib

#endif // include guard
