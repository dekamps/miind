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
#ifndef _CODE_LIBS_DYNAMICNETWORKIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICNETWORKIMPLEMENTATION_INCLUDE_GUARD

#include <string>
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "DynamicNode.h"
#include "SpatialPosition.h"

using std::istream;
using std::ostream;
using std::string;
using std::vector;

using SparseImplementationLib::SparseImplementation;
using SparseImplementationLib::SparseImplementationAllocator;

namespace DynamicLib
{
	//! DynamicNetworkImplementation: an implementation for a DynamicNetwork, which derives from SparseImplementation.

	//! 
	template <class WeightValue>
	class DynamicNetworkImplementation : 
		public SparseImplementation< DynamicNode<WeightValue> >
	{
	public:

		typedef WeightValue WeightType_;
		typedef typename AbstractSparseNode<double,WeightValue>::predecessor_iterator predecessor_iterator;
		typedef DynamicNode<WeightValue> node;
		typedef typename vector< node, SparseImplementationAllocator< node > >::iterator node_iterator;
		typedef typename vector< node, SparseImplementationAllocator< node > >::const_iterator const_node_iterator;
		//! An empty implementation
		DynamicNetworkImplementation();

		//! Read an implementation from a stream
		DynamicNetworkImplementation(istream&);

		//! copy constructor for DynamicNetworkImplementation
		DynamicNetworkImplementation
		(
			const DynamicNetworkImplementation<WeightValue>& 
		);

		DynamicNetworkImplementation<WeightValue>& operator=
		(
			const DynamicNetworkImplementation<WeightValue>&
		);

		~DynamicNetworkImplementation();

		NodeId AddNode(const DynamicNode<WeightValue>&);

		//! make the node, labeled by the first nodeId, input other second Node
		bool MakeFirstInputOfSecond
			(
				NodeId,
				NodeId,
				const WeightValue&
			);

		//! Evolve the node in the implementation over at least a time
		bool Evolve(Time);

		bool UpdateHandler();

		//! Check if a node is purely excitatory/inbitory
		bool IsDalesLawSet() const;

		bool SetDalesLaw(bool);

		void ClearSimulation();

		//! Configure begin, end time, log file, results file, etc.
		bool ConfigureSimulation
		(
			const SimulationRunParameter&
		);

		//! 
		bool AssociateNodePosition
		(
			NodeId,
			const SpatialPosition&
		);

		//!
		bool GetPosition
		(
			NodeId,
			SpatialPosition*
		);

		//! depending on a full state or just a rate request, all DynamicNodes are summoned to deliver a report 
		string CollectReport(ReportType);

		//!
		NodeState State(NodeId) const;

		//!
		virtual bool ToStream	(ostream&) const;

		//!
		virtual bool FromStream	(istream&);

		virtual string Tag() const;

		node_iterator begin(){ return _vector_of_nodes.begin(); }

		node_iterator end  (){ return _vector_of_nodes.end(); }


		//! This gives an iterator to the predecessor connections
		predecessor_iterator begin(NodeId id){ return _vector_of_nodes[id._id_value].begin(); }

		//!
		predecessor_iterator end(NodeId id){return _vector_of_nodes[id._id_value].end(); }
	
	protected:

		using SparseImplementation< DynamicNode<WeightValue> >::_vector_of_nodes;

	private:

		bool _dales_law;

	}; // end of DynamicNetworkImplementation

	//! Standard conversion for operations that depend on efficacy online
	inline double ToEfficacy( double efficacy ) { return efficacy; }

} // end of DynamicLib

#endif // include guard
