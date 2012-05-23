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
#ifndef _CODE_LIBS_NETLIB_SPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_SPARSEIMPLEMENTATIONCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <algorithm>
#include "SparseImplementation.h"
#include "CollectSuccessorInformationCode.h"
#include "LocalDefinitions.h"
#include "ReverseImplementationException.h"
#include "SetIncrementedId.h"
#include "TestReverseImplementationCode.h"

using NetLib::THRESHOLD_VALUE;
using NetLib::THRESHOLD_ID;
using NetLib::D_Pattern;
using NetLib::NetworkParsingException;

namespace SparseImplementationLib
{


	template <class NodeType>
	SparseImplementation<NodeType>::SparseImplementation():
	_b_threshold(false),
	_number_of_input_nodes (0),
	_number_of_output_nodes(0),
	_vector_of_nodes(1) // retain the convention that NodeIds are 1, ...	
	{
		InitializeNodes();
	}

	template <class NodeType>
	SparseImplementation<NodeType>::SparseImplementation
	(
		AbstractArchitecture* p_architecture
	):
	_b_threshold(p_architecture->HaveAThreshold()),
	_number_of_input_nodes(p_architecture->NumberOfInputNodes()),
	_number_of_output_nodes(p_architecture->NumberOfOutputNodes()),
	_vector_of_nodes(InitializeNodeVector(*p_architecture))
	{
		InitializeNodes(p_architecture);

		assert(IsValidNodeArray());
	}

	template <class NodeType>
	SparseImplementation<NodeType>::SparseImplementation
	( 
		const SparseImplementation<NodeType>& rhs 
	):
	_b_threshold(rhs._b_threshold),
	_number_of_input_nodes (rhs._number_of_input_nodes),
	_number_of_output_nodes(rhs._number_of_output_nodes),
	_vector_of_nodes(rhs._vector_of_nodes)
	{

		assert(IsValidNodeArray());
	}

	template <class NodeType>
	SparseImplementation<NodeType>::SparseImplementation(istream& s):
	_b_threshold(false),
	_number_of_input_nodes(0),
	_number_of_output_nodes(0),
	_vector_of_nodes(0)
	{
		// The vector must exist, before it can be parse, otherwise
		// the pointer offsets will be wrong

		if (! ParseVectorOfNodes(s) )
			throw NetworkParsingException(STR_NODE_PARSING_ERROR);

	}

	template <class NodeType>
	inline SparseImplementation<NodeType>::~SparseImplementation()
	{
		assert(IsValidNodeArray());
	}

	template <class NodeType>
	void SparseImplementation<NodeType>::InitializeNodes()
	{
		assert( _vector_of_nodes.size() > 0 );

		// always initialize threshold node
		_vector_of_nodes[THRESHOLD_ID._id_value].SetValue(THRESHOLD_VALUE);

		transform
		(
			_vector_of_nodes.begin(),
			_vector_of_nodes.end(),
			_vector_of_nodes.begin(),
			SetIncrementedId<NodeType>()
		);
	}

	template <class NodeType>
	void SparseImplementation<NodeType>::ExchangeSquashingFunction
	(
		const AbstractSquashingFunction& function
	)
	{
		for
		(
			typename vector<NodeType, SparseImplementationAllocator<NodeType> >::iterator iter = _vector_of_nodes.begin();
			iter != _vector_of_nodes.end();
			iter++
		)
			iter->ExchangeSquashingFunction(&function);

	}

	template <class NodeType>
	bool SparseImplementation<NodeType>::IsValidNodeArray() const
	{
		// Check if the Nodes in this array, have input Node pointers,
		// that also point to Nodes in this array.
		// This is a useful check to see if copy operations work 
		// correctly on an implementation.

		typename vector<NodeType, SparseImplementationAllocator<NodeType> >::const_iterator iter_begin = _vector_of_nodes.begin();
		typename vector<NodeType, SparseImplementationAllocator<NodeType> >::const_iterator iter_end   = _vector_of_nodes.end();

		const NodeType* p_begin = &(*iter_begin);
		const NodeType* p_end   = &(*(iter_end - 1));
	
		// Loop over all Nodes
		for ( typename vector<NodeType, SparseImplementationAllocator<NodeType> >::const_iterator iter = iter_begin; 
				iter != iter_end; 
				iter++ 
			)

			// Check if the Range of the Input Neurons is valid
			if ( ! iter->IsInputNodeInRange(p_begin,p_end) )
				 return false;
	
		return true;
	}

	template <class NodeType>
	SparseImplementation<NodeType>&
		SparseImplementation<NodeType>::operator=
		(
			const SparseImplementation<NodeType>& rhs 
		) 
	{
		if ( this == &rhs )
			return *this;
		_vector_of_nodes        = rhs._vector_of_nodes;

		AdaptNodeArray(rhs);

		assert(IsValidNodeArray());

		_number_of_input_nodes  = rhs._number_of_input_nodes;
		_number_of_output_nodes = rhs._number_of_output_nodes;
	
		return *this;
	}

	template <class NodeType>
	NodeIterator<NodeType> SparseImplementation<NodeType>::begin() 
	{

		// vector starts at zero, but NodeId(0) is threshold node
		NodeType* p_begin_of_evolve = &(_vector_of_nodes[1]);

		// if there are 2 nodes, and we are already at 1, we only have one to go

		NodeIterator<NodeType> order(p_begin_of_evolve);
		return order;
	  }

	template <class NodeType>
	ConstNodeIterator<NodeType> SparseImplementation<NodeType>::begin() const
	{
		// vector starts at zero, but NodeId(0) is threshold node
		const NodeType* p_begin_of_evolve = &(_vector_of_nodes[1]);

		// if there are 2 nodes, and we are already at 1, we only have one to go

		ConstNodeIterator<NodeType> order(p_begin_of_evolve);
		return order;
	}

	template <class NodeType>
	NodeIterator<NodeType> SparseImplementation<NodeType>::end() 
	{	
		// point one past last element of vector, as per convention
		NodeType* p_end_of_evolve  = &(*(--_vector_of_nodes.end()));

		NodeIterator<NodeType> order(++p_end_of_evolve);
		return order;
	}

	template <class NodeType>
	ConstNodeIterator<NodeType> SparseImplementation<NodeType>::end() const
	{
		const NodeType* p_end_of_evolve  = &(*(--_vector_of_nodes.end()));

		ConstNodeIterator<NodeType> order(++p_end_of_evolve);
		return order;
	}

	template <class NodeType>
	inline void SparseImplementation<NodeType>::Insert
	(
		NodeId nid, 
		typename NodeType::ActivityType f_value
	)
	{
		_vector_of_nodes[nid._id_value].SetValue(f_value);
	}

	template <class NodeType>
	inline typename NodeType::ActivityType SparseImplementation<NodeType>::Retrieve(NodeId nid) const
	{
		return _vector_of_nodes[nid._id_value].GetValue();
	}

	template <class NodeType>
	bool SparseImplementation<NodeType>::ReadIn
	(
		const Pattern<typename NodeType::ActivityType>& pat
	)
	{
		assert (pat.Size() == NumberOfInputNodes() );

		for (Index n_pat_index = 0; n_pat_index < pat.Size(); n_pat_index++ )
			_vector_of_nodes[n_pat_index+1].SetValue(pat[n_pat_index]);

		return true;
	}

	template <class NodeType>
	Pattern<typename NodeType::ActivityType> 
		SparseImplementation<NodeType>::ReadOut
	(
	) const
	{

		Pattern<typename NodeType::ActivityType> pat_ret(NumberOfOutputNodes());

		size_t n_start_index = NumberOfNodes() - NumberOfOutputNodes();
	
		for (Index n_pat_index = 0; n_pat_index < NumberOfOutputNodes(); n_pat_index++ )
		{ 
			size_t vec_index = n_pat_index + n_start_index + 1;
			pat_ret[n_pat_index] = _vector_of_nodes[vec_index].GetValue();
		}

		return pat_ret;
	}

	template <class NodeType>
	bool SparseImplementation<NodeType>::ToStream
	(
		ostream& s
	) const
	{	
		// When called from a derived class, this function should have the correct type
		s << SparseImplementation<NodeType>::Tag() << "\n";

		s << STR_THRESHOLD_TAG           << " "  <<
			 (_b_threshold ? 1 : 0)      << " "  << 
			 ToEndTag(STR_THRESHOLD_TAG) << " "  <<
			 NumberOfNodes()             << " "  << 
			 NumberOfInputNodes()        << " "  << 
			 NumberOfOutputNodes()       << "\n";

		copy
		(
			_vector_of_nodes.begin(),
			_vector_of_nodes.end(),
			ostream_iterator<NodeType>(s)
		);


		s << ToEndTag(SparseImplementation<NodeType>::Tag()) << "\n";

		return true;
	}

	template <class NodeType>
	bool SparseImplementation<NodeType>::FromStream(istream& s)
	{
		if (! ParseVectorOfNodes(s) )
			throw NetworkParsingException(string("Couldn't parse node vector"));
		return true;
	}

	template <class NodeType>
	string SparseImplementation<NodeType>::Tag() const
	{
		return STR_SPARSEIMPLEMENTATION_TAG;
	}


	template <class NodeType>
	void SparseImplementation<NodeType>::AddPredecessor
	(
		NodeId ThisId, 
		NodeId PredecId
	)
	{
		NodeType* p_neur = &_vector_of_nodes[PredecId._id_value];
		pair< NodeType*, WeightValue>  con(p_neur,1);
		_vector_of_nodes[ThisId._id_value].PushBackConnection(con);
	}

	template <class NodeType>
	void SparseImplementation<NodeType>::InitializeNodes
	(
		AbstractArchitecture* p_architecture
	)
	{
		NodeLink nlink;

		while ( p_architecture->Collection()->NumberOfNodes())
		{
			nlink = p_architecture->Collection()->pop();

			if (p_architecture->HaveAThreshold())

			// every neuron, including the input neurons, have the threshold neuron
			// as predecessor, if the architecture says so

				AddPredecessor(nlink.MyNodeId(), NodeId(0));	


			Number nr_of_predecessors = nlink.Size();
			for ( Index n_index = 0; n_index < nr_of_predecessors; n_index++ )
				AddPredecessor( nlink.MyNodeId(), nlink[n_index] );

		}
	}

	template <class NodeType>
	vector<NodeType, SparseImplementationAllocator<NodeType> > SparseImplementation<NodeType>::InitializeNodeVector
	(
		const AbstractArchitecture& architecture
	)
	{
		vector<NodeType, SparseImplementationAllocator<NodeType> > vector_return(architecture.NumberOfNodes() + 1);

		transform
		(		
			vector_return.begin(),
			vector_return.end(),
			vector_return.begin(),
			SetIncrementedId<NodeType>()
		);

		vector_return[THRESHOLD_ID._id_value].SetValue(THRESHOLD_VALUE);

		return vector_return;
	}

	template <class NodeType>
	NodeLinkCollection 
		SparseImplementation<NodeType>::FromNeurVecToCollection
		(
			const vector<NodeType, SparseImplementationAllocator<NodeType> >& vec_neur
		)
	{

		// create a vector of links
		vector<NodeLink> vec_link;

		// loop over  all neurons in the vector
		vector<NodeLink> vec_links;
		size_t nr_neurons = vec_neur.size();
		size_t n_neur_ind;
		for ( n_neur_ind = 0; n_neur_ind < nr_neurons; n_neur_ind++ )
		{
			// loop over all input neurons
			size_t n_input_ind;
			size_t nr_input = vec_neur[n_neur_ind].NrInput();
			NodeId my_id = vec_neur[n_neur_ind].MyNeuronId();
			vector<NodeId> vec_input;

			for ( n_input_ind = 0; n_input_ind < nr_input; n_input_ind++ )
			{
				// look for input id  and push it on the input vector
				// if it is not id 0 ( threshold id is implicitely assumed by NeuroLinkCollection)
				NodeId id_predec = vec_neur[n_neur_ind].IdPredecessor( n_input_ind );
				if ( id_predec._id_value != 0)
					vec_input.push_back( id_predec );
			}

			// create a link and push the link on the link vector
			NodeLink Link(my_id,vec_input);
			vec_links.push_back(Link);
		}

		// make the neuronlinkcollection and return it
		NodeLinkCollection LinkCol( vec_links );
		return LinkCol;

	}

	template <class NodeType>
	bool SparseImplementation<NodeType>::AdaptNodeArray(const SparseImplementation<NodeType>& rhs) const
	{
		// Use this to copy arrays of sparse neurons
		// Tested: 08-12-1999
		// Author: Marc de Kamps
		// Modified: 13-11-2003, calculate a single offset between the starting element of this and rhs and
		// pass this to a SparseNode to calculate the new connection pointers
		// Modified: 22-02-2006, previous method was flaky. Simply pass the two array elements pointers to the new
		// node and let it calculate the new pointers from the offsets
	
		typedef AbstractSparseNode<typename NodeType::ActivityType,typename NodeType::WeightType>* node_pointer;
		typedef const AbstractSparseNode<typename NodeType::ActivityType,typename NodeType::WeightType>* const_node_pointer; 

		typename vector<NodeType,SparseImplementationAllocator<NodeType> >::const_iterator iter_begin = _vector_of_nodes.begin();
		typename vector<NodeType,SparseImplementationAllocator<NodeType> >::const_iterator iter_end   = _vector_of_nodes.end();
	
		for 
		(	
			typename vector<NodeType,SparseImplementationAllocator<NodeType> >::const_iterator iter = iter_begin; 
			iter != iter_end; 
			iter++
		)
		{

			const_node_pointer p_right = &(*rhs._vector_of_nodes.begin());
			node_pointer p_old = const_cast<node_pointer>(p_right)->Address(iter - iter_begin);
	
			iter->ApplyOffset
			(
				const_cast<const_node_pointer>(p_old)
			);

		}

		return true;
	}

	template <class NodeType>
	inline bool SparseImplementation<NodeType>::InsertWeight
	( 
		NodeId Out, 
		NodeId In, 
		typename NodeType::WeightType w_value 
	)
	{
		NodeType& out = _vector_of_nodes[Out._id_value];
		typename NodeType::predecessor_iterator iter_predecessor =
			find
			(
				out.begin(),
				out.end(),
				In
			);

		if ( iter_predecessor != out.end() )
		{
			iter_predecessor.SetWeight( w_value );
			return true;
		}
		else
			return false;
	}

	template <class NodeType>
	inline bool SparseImplementation<NodeType>::GetWeight
			(
				NodeId Out, 
				NodeId In, 
				typename NodeType::WeightType&  weight_value
			) const
	{
		NodeType out = _vector_of_nodes[Out._id_value];
		typename NodeType::predecessor_iterator iter_predecessor =
			find
			(
				out.begin(),
				out.end(),
				In
			);

		if ( iter_predecessor != out.end() )
		{
			weight_value = iter_predecessor.GetWeight();
			return true;
		}
		else
			return false;

	}

	template <class NodeType>
	inline Number SparseImplementation<NodeType>::NumberOfNodes() const
	{
		assert(_vector_of_nodes.size() >0);
		return static_cast<Number>(_vector_of_nodes.size()) - 1;
	}

	template <class NodeType>
	inline Number SparseImplementation<NodeType>::NumberOfInputNodes() const
	{
		return _number_of_input_nodes;
	}

	template <class NodeType>
	inline Number SparseImplementation<NodeType>::NumberOfOutputNodes() const
	{
		return _number_of_output_nodes;
	}

	template <class NodeType>
	void SparseImplementation<NodeType>::ScaleWeights(WeightValue scale)
	{
		typedef NodeIterator<NodeType> NodeIter;
		for( NodeIter iter = this->begin(); iter != this->end(); iter++)
			iter->ScaleWeights(scale);
	}

	template <class NodeType>
	bool SparseImplementation<NodeType>::ParseVectorOfNodes(istream& s)
	{
		string str_header;
		s >> str_header;
		if (str_header != Tag() )
			throw NetworkParsingException(string("Expected sparse implementation header"));
	
		s >> str_header;
		if (str_header != STR_THRESHOLD_TAG)
			throw NetworkParsingException(STR_THRESHOLD_TAG_EXPECTED);
		int i_threshold;
		s >> i_threshold;
		_b_threshold = i_threshold ? true: false;
		s >> str_header;
		if ( str_header != ToEndTag(STR_THRESHOLD_TAG) )
			throw NetworkParsingException(STR_THRESHOLD_TAG_EXPECTED);

		int number_nodes;
		s >> number_nodes >> _number_of_input_nodes >> _number_of_output_nodes;

		// Now, the element that is in _vector_of_nodes that is being read in
		// must exist. Only now we know how many Nodes there are, so there
		// was no way to create the entire vector. 
		vector<NodeType, SparseImplementationAllocator<NodeType> > vector_dummy(number_nodes + 1);
		_vector_of_nodes = vector_dummy; 

		// remember threshold Node, i.e. range is from zero to <= number Nodes
		for (int node_index = 0; node_index <= number_nodes; node_index++)
			_vector_of_nodes[node_index].FromStream(s);
	
		
		string str_footer;
		s >> str_footer;
		if (str_footer != ToEndTag(Tag()))
			throw NetworkParsingException(string("Expected sparse implementation footer"));

		return true;
	}

	template <class NodeType>
	inline bool SparseImplementation<NodeType>::IsInputNeuron(NodeId nid) const
	{
		if  ( static_cast<size_t>(nid._id_value) <= NumberOfInputNodes() )
			return true;
		else 
			return false;
	}

	template <class NodeType>
	bool SparseImplementation<NodeType>::InsertReverseImplementation()
	{	
		// create helper object
		NodeLinkCollectionFromSparseImplementation<NodeType> collection;
		NodeLinkCollection reverse_collection = collection.CreateReverseNodeLinkCollection(_vector_of_nodes);

		CollectSuccesorInformation<NodeType> add_reverse_connections
							(
								_b_threshold,
								reverse_collection,
								_vector_of_nodes,
								NumberOfInputNodes()
							);
		transform
		(
			_vector_of_nodes.begin(), 
			_vector_of_nodes.end(), 
			_vector_of_nodes.begin(), 
			add_reverse_connections
		);

		return true;
	}
		
	template <class NodeType>
	bool SparseImplementation<NodeType>::IsReverseImplementationConsistent() 
	{
		typename vector<NodeType, SparseImplementationAllocator<NodeType> >::iterator iter_begin = _vector_of_nodes.begin();
		typename vector<NodeType, SparseImplementationAllocator<NodeType> >::iterator iter_end   = _vector_of_nodes.end();
		
		TestReverseImplementation<NodeType,WeightValue> test(_vector_of_nodes);

		try
		{
			std::for_each
				(
					iter_begin, 
					iter_end,
					test
				);
		}
		catch(ReverseImplementationException)
		{
			throw ReverseImplementationException(string("Couldn't collect Reverse information"));
		}

		return true;
	}	
	
	template <class NodeType>
	NodeType& SparseImplementation<NodeType>::Node(NodeId id)
	{
	        return _vector_of_nodes[id._id_value];
	}


	template <class NodeType>
	const NodeType& SparseImplementation<NodeType>::Node(NodeId id) const
	{
	        return _vector_of_nodes[id._id_value];
	}      

} // end of SparseImplementationLib

template <class NodeType>
ostream& SparseImplementationLib::operator<<
(
	ostream& s, 
	const SparseImplementation<NodeType>& implementation
)
{
	implementation.ToStream(s);
	return s;
}	

#endif // include guard
