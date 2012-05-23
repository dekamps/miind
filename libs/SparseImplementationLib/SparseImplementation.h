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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEIMPLEMENTATION_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <utility>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include "../UtilLib/UtilLib.h"
#include "../NetLib/NetLib.h"
#include "AbstractSparseNodeCode.h"
#include "BasicDefinitions.h"
#include "SparseImplementationAllocator.h"
#include "SparseNodeCode.h"
#include "ReversibleSparseNodeCode.h"
#include "LayerWeightIteratorThresholdCode.h"
#include "LayerWeightIterator.h"
#include "NodeLinkCollectionFromSparseImplementationCode.h"

using std::auto_ptr;
using NetLib::AbstractSquashingFunction;
using NetLib::NodeIterator;
using NetLib::ConstNodeIterator;
using NetLib::AbstractArchitecture;
using NetLib::Pattern;
using NetLib::Layer;
using UtilLib::Streamable;

namespace SparseImplementationLib {

	//! This is a collection of SparseNode s and thereby an explicit representation of a sparse network.
	//! Usually this class is used as implementation in conjunction with an interface class. This representation
	//! is optimized for representations of sparse irregular networks in a single threaded program.

	template <class NodeType_>
	class SparseImplementation : public Streamable 	{
	public:

		//! allows to deduce NodeType from implementation template argument
		typedef NodeType_ NodeType;
		
		//! allows to deduce type of Nodes activation value and of weights
		typedef typename NodeType_::ActivityType NodeValue;
		typedef typename NodeType_::WeightType   WeightValue;
		//!
		typedef SparseImplementation<NodeType>& implementation_reference;

		//! Default excution order given by LayerOrder 
		typedef NodeIterator<NodeType> Order;

		//! Empty implementation
		SparseImplementation();

		//! Create an implementation directly from a stream
		SparseImplementation(istream&);

		//! Create an implementation from an Architecture
		SparseImplementation(AbstractArchitecture*);

		//! Copy constructor
		SparseImplementation(const SparseImplementation<NodeType>&);

		//! Derived classes expected
		virtual ~SparseImplementation();

		//! copy operator
		SparseImplementation&	operator=( const SparseImplementation<NodeType>& );

		//! Generates a Pattern from the activity values present in the output nodes.
		Pattern<NodeValue> ReadOut() const;

		//! iterator which points to begin Node in the implementation
		NodeIterator<NodeType> begin ();

		//! iterator which points to one past the end Node in the implementation
		NodeIterator<NodeType> end   ();

		ConstNodeIterator<NodeType> begin() const;

		ConstNodeIterator<NodeType> end() const;

		NodeType& Node(NodeId id);

		const NodeType& Node(NodeId id) const;

		// Insertion methods:
		bool ReadIn(const Pattern<NodeValue>& );

		void Insert
			(
				NodeId, 
				NodeValue
			);
		bool InsertWeight
			(
				NodeId, 
				NodeId, 
				WeightValue
			);	//Robust, but inefficient

		NodeValue Retrieve(NodeId)          const;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
		bool	GetWeight
			(
				NodeId, 
				NodeId, 
				WeightValue&
			) const;
		
		//! in all nodes, exchange the current squashing function by a new one
		void ExchangeSquashingFunction
			(
				const AbstractSquashingFunction&
			);
		//! put an implementation to a stream
		virtual bool ToStream   (ostream&) const;

		//! build an implementation from a stream
		virtual bool FromStream (istream&);
	
		//!
		virtual string Tag() const;

		//! Network property functions:
		Number NumberOfNodes       () const;
		Number NumberOfInputNodes  () const;
		Number NumberOfOutputNodes () const;


		bool InsertReverseImplementation();

		//! this will insert reverse connections in all nodes if they are Reversible,
		//! if they are not the compiler will remind you by generating an error for this function
		bool IsReverseImplementationConsistent();

		//! Scale all weights by some factor
		void ScaleWeights(WeightValue);

	protected:

	private:

		void AddPredecessor
		(
			NodeId, 
			NodeId
		);

		vector<NodeType, SparseImplementationAllocator<NodeType> >	
			InitializeNodeVector
			(
				const AbstractArchitecture&
			);

		bool	AdaptNodeArray
		(
			const SparseImplementation<NodeType_>&
		) const;

		bool	IsValidNodeArray() const;

		void	InitializeNodes();

		void	InitializeNodes
		(
			AbstractArchitecture*
		);

		bool	InitializeBaseImp
		(
			istream&
		);

		NodeLinkCollection  ReverseLinkCollection() const;
		NodeLinkCollection  FromNeurVecToCollection(const vector<NodeType, SparseImplementationAllocator<NodeType> >&);


		bool SetValues(const vector<NodeType>&);
		bool ParseVectorOfNodes(istream&);

		bool			_b_threshold;

		Number	_number_of_input_nodes;   // Number of input Nodes
		Number	_number_of_output_nodes;  // Number of output Nodes

	protected:

		bool IsInputNeuron(NodeId) const;

		vector<NodeType,SparseImplementationAllocator<NodeType>	> _vector_of_nodes;	      // vector of network Nodes

	};

	template <class NodeType>
	ostream& operator<<( std::ostream&, const SparseImplementation<NodeType>& );
	template <class NodeType>
	istream& operator>>( std::istream&, SparseImplementation<NodeType>& );

	template <class NodeType>
	struct T_SparseImplementation
	{
		typedef SparseImplementation<NodeType>  Type;
	};

	typedef SparseImplementation< SparseNode<double,double> >           D_SparseImplementation;
	typedef SparseImplementation< ReversibleSparseNode<double,double> > D_ReversibleSparseImplementation;
	// inserted by korbo
	typedef SparseImplementation< SparseNode<float,float> >           F_SparseImplementation;
	typedef SparseImplementation< ReversibleSparseNode<float,float> > F_ReversibleSparseImplementation;

} // end of SparseImplementationLib


#endif // include guard
