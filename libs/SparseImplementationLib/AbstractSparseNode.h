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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_ABSTRACTSPARSENODE_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_ABSTRACTSPARSENODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "../NetLib/NetLib.h"
#include "ConnectionProduct.h"
#include "LocalDefinitions.h"
#include "SparseLibException.h"
#include "ConstSparsePredecessorIterator.h"
#include "SparsePredecessorIterator.h"

using std::istream;
using std::ostream;
using NetLib::NodeId;
using UtilLib::Index;
using UtilLib::NumericException;
using UtilLib::Streamable;

namespace SparseImplementationLib
{
	template <class N> class SparseImplementationAllocator;

	template <class A, class W> class AbstractSparseNode;

	template <class ActivityType, class WeightType>
	ostream& operator<<
	(
		ostream&, 
		const AbstractSparseNode<ActivityType,WeightType>&
	);

	//! A class for representing nodes in sparse networks.

	//! The key idea of an AbstractSparseNode is that sparse networks can be represented efficiently as follows:
	//! <ul>
	//! <li>Create a collection of nodes</li>
	//! <li>Give each node a pointer-weight pair</li>
	//! </ul>
	//! In general networks are represented by a collection of nodes and an adjecency matrix which records, which nodes
	//! are connected to each other and how strongly. In sparse networks most elements of this adjacency matrix are zero.
	//! A lage matrix full of zeroes is inefficiecient, so if it is known beforehand that the network is sparse, equivalent
	//! network representations must be sought which are more efficient. One way of doing this is, is to make the node itself
	//! responsible for representing its neighbour nodes. So, each node maintains a list of its 'predecessors'. In C++ terms:
	//! each node maintains a vector ('list') of pointer-weight pairs. The pointer is a pointer to other C++ nodes, which are also
	//! part of the network. The weight determines the efficacy of the connection.  A collection of such nodes thus represents a 
	//! network. 
	//!
	//! \section Usage
	//! AbstractSparseNode is an abstract base class, due to its pure virtual destructor. A concrete node type that can be instantiated
	//! is the SparseNode. It is recommended to study usage there.
	//!
	//! \section Design
	//! The original design anticipated usage of nodes whose activities could be represented by different type such as
	//! bool, e.g. for Hopfield networks, float for normal neural networks where precision is not paramount, double etc.
	//! Hence, ActivityType is a template parameter that sets the type of the nodes activation value. Similarly different types
	//! for weights could be anticipated, which can be set independently by the WeightType argument.  A node should uniquely defined
	//! by its NodeId (it is the responsibility for a SparseImplementation to ensure that this is the case. The node has an _activation_value
	//! field for its activation value. A connection is a pair of node pointers and weights. A connection can be added with PushBackConnection method.
	//! Once connections have been added to the node, predecssor_iterator can be used to iterate over the connections (and therefore the predecessors of a given node.
	//!
	//! \section Special Considerations
	//! AbstractSparseNode was intended as a base class, providing basic services for more complex node types, such as for example, DynamicNode.
	//! The original idea was that the tricky aspects of the pointer usages for the representation of this connection would be handled
	//! by this base class. One example is the copying of networks, where pointer-weight pairs must be updated so that pointers in the
	//! new version of the network point to nodes in the new networks and to nodes in the old network. This has mostly worked: the
	//! intracies of network copying and serialization are handled successfully by SparseImplementation. However, succesfully implemented
	//! derived classes of AbstractSparseNode must take into account that for their connection representation they rely on the AbstractSparseNode
	//! base class and that the pointers in this derived class are of type AbstractSparseNode*. This means that pointer arithmetic
	//! on these pointers is inconsistent, if it is assumed that the pointers are to type DerivedClass*.
	//! To make sure that SparseImplementation correctly copies and serializes networks of such derived classes, the Address and Offset 
	//! methods must be overloaded. The documentation for these functions explains how this must be done.
	//! Concrete examples are provided by SparseNode and DynamicNode.
	//!
	//! The original design motivation was the possibility to create inhomogenous networks. DynamicNode shows how polymorphic node
	//! behaviour is achieved with a homogeneous set of DynamicNode. TmeplatedNode is in preparation to superseed AbstractSparseNode
	//! in most practical applications.

	template <class ActivityType_, class WeightType_>
	class AbstractSparseNode : public Streamable
	{
	public:

		typedef AbstractSparseNode<ActivityType_,WeightType_>*				node_pointer;
		typedef const AbstractSparseNode<ActivityType_, WeightType_>*		const_node_pointer;
		typedef pair<node_pointer,WeightType_>								connection;
		typedef typename vector<connection>::const_iterator					connection_iterator;
		typedef SparsePredecessorIterator<AbstractSparseNode<ActivityType_,WeightType_> >		
																			predecessor_iterator;
		typedef ConstSparsePredecessorIterator<AbstractSparseNode<ActivityType_,WeightType_> >
																			const_predecessor_iterator;


		typedef ActivityType_	ActivityType;
		typedef WeightType_		WeightType;

		//! Standard constructor, sets a NodeId
		AbstractSparseNode(NodeId);

		//! copy constructor
		AbstractSparseNode(const AbstractSparseNode<ActivityType, WeightType>& );

		//! virtual destructor
		virtual ~AbstractSparseNode() = 0;
	
		//! copy operator
		AbstractSparseNode<ActivityType,WeightType>& 
			operator=(const AbstractSparseNode<ActivityType,WeightType>&);

		//! virtual output streaming function 
		virtual bool ToStream   (ostream&) const;

		//! virtual input streaming function
		virtual bool FromStream (istream&);
		
		//! Object tag
		virtual string Tag() const;

		//! Set the value (numererical value) of the node
		void SetValue(ActivityType);

		//! Get the value
		ActivityType	GetValue()       const;

		//! Calculate InnerProduct over input values
		double InnerProduct   ()   const;

		//! Get NodeId
		NodeId MyNodeId() const { return _my_id; } 

		//! Set the current Node's Id value
		void SetId(NodeId);

		//! Push back a connection (to one of the node's predecessors)
		void PushBackConnection
		(
			const connection& //!< a connection is a pair of pointer to a node and a weight
		);
        
		//! iterator to connections: begin. If there are no input connections, dereference operations are illegal, this is when begin() == end()
		predecessor_iterator begin ();

		//! iterator to connections: end
		predecessor_iterator end   ();

		//! const iterator to connections: begin. If there are no input connections, dereference operations are illegal, this is when begin() == end()
		const_predecessor_iterator begin() const;

		//! const iterator to connections: end
		const_predecessor_iterator end() const;

		template <class N> friend class SparseImplementationAllocator;

		//! scale all weight in the connection of this node
		void ScaleWeights(WeightType); 

	public:
		
		virtual node_pointer Address(std::ptrdiff_t) = 0;

		virtual std::ptrdiff_t Offset(node_pointer) const = 0;

		void Clear();

	public:

		void ApplyOffset
		(
			const_node_pointer
		) const ;


		// Test to see if the input Nodes of this Neuron
		// belong to a 'network'
		bool IsInputNodeInRange
		(
			const_node_pointer, 
			const_node_pointer
		) const;

	private:




		ActivityType	_activation_value;
		NetLib::NodeId	_my_id;

		// deep copy requires pointers be changeable
		mutable vector<connection> _vector_of_connections;


	}; // end of AbstractSparseNode


	//! All derived classes from AbstractSparseNode can use operator>>
	template <class ActivityType, class WeightType>
	istream& operator>>(istream&, AbstractSparseNode<ActivityType,WeightType>&);

	template <class ActivityType, class WeightType>
	bool operator==
	( 
		const AbstractSparseNode<ActivityType, WeightType>&, 
		const AbstractSparseNode<ActivityType, WeightType>& 
	);
	

	template <class NodeValue, class WeightValue>
	bool operator==( const AbstractSparseNode<NodeValue,WeightValue>& node, NodeId id )
	{
		return (node.MyNodeId() == id);
	}


	template <class ActivityType, class WeightType>
	ostream& operator<<
	(
		ostream&, 
		const pair<AbstractSparseNode<ActivityType,WeightType>*, WeightType >&
	);

	template <class NodeValue, class WeightValue>
	ostream& operator<<(ostream& s, const pair<AbstractSparseNode<NodeValue,WeightValue>*, WeightValue >& connection)
	{
	  //TODO: g++ doesn't accept IsNan(OU)
	  //		if ( IsNan(connection.second) )
	  //	throw NumericException(INVALID_WEIGHT);

		s << connection.first->MyNodeId() << " " <<  connection.second << " ";
		return s;
	}
	typedef AbstractSparseNode<double, double> D_AbstractSparseNode;



} // end of SparseImplementationLib

#endif // include guard
