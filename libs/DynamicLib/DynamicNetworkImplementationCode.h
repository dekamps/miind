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

#ifndef _CODE_LIBS_DYNAMICNETWORKIMPLEMENTATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICNETWORKIMPLEMENTATIONCODE_INCLUDE_GUARD

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include "DynamicNetworkImplementation.h"
#include "ConfigureNodes.h"
#include "EvolveNodes.h"
#include "IterationNumberException.h"
#include "ReallocException.h"

namespace DynamicLib
{
	template <class WeightValue>
	DynamicNetworkImplementation<WeightValue>::DynamicNetworkImplementation():
	SparseImplementation< DynamicNode<WeightValue> >(),
	_dales_law(true)
	{
	}

	template <class WeightValue>
	  DynamicNetworkImplementation<WeightValue>::DynamicNetworkImplementation(std::istream&):
	SparseImplementation< DynamicNode<WeightValue> >()
	  {
	  }

	template <class WeightValue>
	DynamicNetworkImplementation<WeightValue>::DynamicNetworkImplementation
	(
		const DynamicNetworkImplementation<WeightValue>& rhs
	):
	SparseImplementation< DynamicNode<WeightValue> >
	(
		rhs
	),
	_dales_law(rhs._dales_law)
	{
	}

	template <class WeightValue>
	DynamicNetworkImplementation<WeightValue>::~DynamicNetworkImplementation()
	{
	}

	template <class WeightValue>
	bool DynamicNetworkImplementation<WeightValue>::FromStream(istream& s)
	{
		typedef typename DynamicNetworkImplementation<WeightValue>::NodeType NodeType;
		string dummy;

		s >> dummy;
		if (dummy != this->Tag() )
			throw DynamicLibException("Unexpected DynamicNetworkImplementation tag");
		Number n_nodes;
		s >> n_nodes;

		_vector_of_nodes.resize(n_nodes);
		BOOST_FOREACH(NodeType& node, _vector_of_nodes)
			node.FromStream(s);

		s >> dummy;
		if (dummy != this->ToEndTag(this->Tag()) )
			throw DynamicLibException("Unexpected DynamicNetworkImplementation tag");

		return true;
	}

       	template <class WeightValue>
	bool DynamicNetworkImplementation<WeightValue>::ToStream(ostream& s) const
	{
		s << Tag() << "\n";

		s << _vector_of_nodes.size() << endl;
      	for (const_node_iterator iter = _vector_of_nodes.begin(); iter != _vector_of_nodes.end();iter++)
         	iter->ToStream(s);

		s << ToEndTag(Tag()) << "\n";

		return true;
	}

	template <class WeightValue>
	string DynamicNetworkImplementation<WeightValue>::Tag() const
	{
		return STR_DYNAMICNETWORKIMPLEMENTATION_TAG;
	}

	template <class WeightValue>
	NodeId DynamicNetworkImplementation<WeightValue>::AddNode
	(
		const DynamicNode<WeightValue>& node
	)
	{

		_vector_of_nodes.push_back(node);

		int index_of_new_node = static_cast<int>(_vector_of_nodes.size())-1; 
		_vector_of_nodes[index_of_new_node].SetId(NodeId(index_of_new_node));
		return NodeId(index_of_new_node);
	}

	template <class WeightValue>
	bool DynamicNetworkImplementation<WeightValue>::MakeFirstInputOfSecond
	(
		NodeId id_input,
		NodeId id_receiver,
		const WeightValue& weight
	)
	{
		// See if the Ids make sense
		if ( 
			id_input._id_value    > static_cast<int>(this->NumberOfNodes()) ||
			id_receiver._id_value > static_cast<int>(this->NumberOfNodes()) 
		)
			return false;

		// See if the weights are consistent with the type of the Nodes (Dale's law)
		if ( 
				IsDalesLawSet() &&
				(
					(_vector_of_nodes[id_input._id_value].Type() == EXCITATORY && (ToEfficacy(weight) < 0)) ||
					(_vector_of_nodes[id_input._id_value].Type() == INHIBITORY && (ToEfficacy(weight) > 0))
				)
		)
			throw DynamicLibException("Dale's law violated");

		// everything allright, prepare a connection
		typedef pair<DynamicNode<WeightValue>*,WeightValue> Connection;

		Connection connection;

		connection.first  = &_vector_of_nodes[id_input._id_value];
		connection.second = weight;

		_vector_of_nodes[id_receiver._id_value].PushBackConnection(connection);

		return true;

	}

	template <class WeightValue>
	bool DynamicNetworkImplementation<WeightValue>::Evolve(Time time_to_achieve)
	{
		typedef typename DynamicNetworkImplementation<WeightValue>::NodeType NodeType;

		BOOST_FOREACH(NodeType& node, _vector_of_nodes)
			node.CollectExternalInput();

	
		BOOST_FOREACH(NodeType& node, _vector_of_nodes)
			node.Evolve(time_to_achieve);

		return true;
	}

	template <class WeightValue>
	bool DynamicNetworkImplementation<WeightValue>::ConfigureSimulation
	(
		const SimulationRunParameter& parameter
	)
	{
		ConfigureNodes<WeightValue> configure(parameter);
		
		bool b_result = std::for_each
			(
				_vector_of_nodes.begin(),
				_vector_of_nodes.end(),
				configure
			).Result();
		
		return b_result;
	}

	template <class WeightValue>
	void DynamicNetworkImplementation<WeightValue>::ClearSimulation()
	{
		for_each
		(
			_vector_of_nodes.begin(),
			_vector_of_nodes.end(),
			mem_fun_ref(&DynamicNode<WeightValue>::ClearSimulation)
		);
	}

	template <class WeightValue>
	bool DynamicNetworkImplementation<WeightValue>::UpdateHandler()
	{
		for_each
		(
			_vector_of_nodes.begin(),
			_vector_of_nodes.end(),
			mem_fun_ref(&DynamicNode<WeightValue>::UpdateHandler)
		);

		return true;
	}

	template <class WeightValue>
	NodeState DynamicNetworkImplementation<WeightValue>::State(NodeId id) const
	{
		assert( id._id_value >= 0 && id._id_value < static_cast<int>(_vector_of_nodes.size()) );
		return _vector_of_nodes[id._id_value].State();
	}

	template <class WeightValue>
	string DynamicNetworkImplementation<WeightValue>::CollectReport(ReportType type) 
	{
		node_iterator iter_begin = this->begin();
		node_iterator iter_end   = this->end();
		
		string string_return;
		for
		(
			node_iterator iter = iter_begin;
			iter != iter_end;
			iter++
		)
			string_return = iter->ReportAll(type);

		return string_return;

	}

	template <class Implementation>
	bool DynamicNetworkImplementation<Implementation>::IsDalesLawSet() const
	{
		return _dales_law;
	}

	template <class Implementation>
	bool DynamicNetworkImplementation<Implementation>::SetDalesLaw(bool b_law)
	{
		_dales_law = b_law;
		return b_law;
	}

	template <class Implementation>
	bool DynamicNetworkImplementation<Implementation>::AssociateNodePosition
	(
		NodeId id,
		const SpatialPosition& position
	)
	{
		if 
		( 
			id._id_value > 0 && 
			id._id_value <= static_cast<int>(this->NumberOfNodes()) 
		)
		{
			_vector_of_nodes[id._id_value].AssociatePosition(position);
			return true;
		}
		else
			return false;
	}

	template <class Weight>
	bool DynamicNetworkImplementation<Weight>::GetPosition
	(
		NodeId id,
		SpatialPosition* p_pos
	)
	{
		if (id._id_value < static_cast<int>(_vector_of_nodes.size()) )
		{
			return _vector_of_nodes[id._id_value].GetPosition(p_pos);
		}
		else
			return false;
	}

} // end of DynamicLib



#endif // include guard

