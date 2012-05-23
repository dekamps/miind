// Copyright (c) 2005 - 2008 Marc de Kamps
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


#include "../UtilLib/UtilLib.h"
#include "WeightMatrixElementImplementation.h"
/*
using namespace Util;
using namespace NetLib;
using namespace ImplementationLib;


template <> LayerElement<double,double>::LayerElement( size_t n_input, 
						       size_t n_output ):
_vec_neur( std::vector<double>(n_input + 1) ),
_mat_weights(n_output, n_input + 1)
{

	// setting of threshold neuron

	_vec_neur[0] = THRESHOLD_VALUE;

}

template <> LayerElement<double,double>::LayerElement( const LayerElement<double,double>& rhs ):
_vec_neur( rhs._vec_neur ),
_mat_weights( rhs._mat_weights)
{
}

template <> LayerElement<double,double>::~LayerElement()
{
}

template <> LayerElement<double,double>& 
LayerElement<double,double>::operator=( const LayerElement<double,double>& rhs )
{
	if ( this == &rhs )
		return *this;

	_vec_neur		= rhs._vec_neur;
	_mat_weights	= rhs._mat_weights;

	return *this;
}




template <> void LayerElement<double,double>::Insert( ElementId Eid, double val )
{
	_vec_neur[Eid.value] = val;
}

template <> Number LayerElement<double,double>::NrInput() const
{
	return static_cast<Number>(_mat_weights.NrYdim());
}


template <> double LayerElement<double,double>::Update( struct ElementId  el_id, NodeId nid) const
{
	// Calculate the weighted sum of inputs for this neuron
	// Tested:	29-07-1999
	// By:		Marc de Kamps

	// intialize a sum to zero

	double sum = 0;

	// loop over the number of input neurons

	size_t n_neuron_index;

	for ( n_neuron_index = 0; n_neuron_index < NrInput(); n_neuron_index++ )

		// add matrix(ElementId, neuron_id)*vector(neuron_id) to the sum

		sum += _mat_weights( el_id.value, n_neuron_index)*
			_vec_neur[n_neuron_index];

	// return the sum

	return sum;
	
}


template <> Number LayerElement<double,double>::NrOutput() const
{
	return static_cast<Number>(_mat_weights.NrXdim());
}

template <> double LayerElement<double,double>::Retrieve( ElementId Eid ) const
{
	return _vec_neur[Eid.value];
}

template <> double LayerElement<double,double>::RetrieveWeight( ElementId eid_out, ElementId eid_in ) const
{
	return _mat_weights( eid_out.value, eid_in.value );
}

template <> void LayerElement<double,double>::InsertWeight( ElementId eid_out, ElementId eid_in, double val )
{
	_mat_weights(eid_out.value,eid_in.value) = val;
}




template <> std::vector< LayerElement<double,double>* >
	WeightMatrixElementImplementation<double,double>::InitializeElementVector( Architecture& Arch )
{
	// Create return vector

	std::vector< LayerElement<double,double>* > vec_ret;

	// Loop over all layers but the last one

	LayerElement<double,double>* p_element; 
	Index n_layer_index;
	for ( n_layer_index = 0; n_layer_index < Arch.NrLayers() - 1; n_layer_index++ )
	{

		// Create an element with input number:  the number of neurons in this layer
		//					 with output number: the number of neurons in the next layer

		p_element = new LayerElement<double,double>(Arch[n_layer_index],
													Arch[n_layer_index+1] );

		// push back the pointer to the element on the vector

		vec_ret.push_back(p_element);

	}

	// Create one final output element ( 0 output neurons )

	p_element = new LayerElement<double,double>(Arch[Arch.NrLayers() - 1], 0  );

	// push it back on the vector
	
	vec_ret.push_back( p_element );

	// return the vector

	return vec_ret;
}


template<> std::vector<size_t>
	WeightMatrixElementImplementation<double,double>::InitializeAccumulativeVector( Architecture& Arch )
{
	// Sets each vector element to the first NeuronId of each  layer.
	// Tested:	26-7-1999
	// By:		Marc de Kamps

	std::vector<Number> vec_ret(Arch.NrLayers() + 1);

	Number n_sum = 1;
	Index n_index;

	for ( n_index = 0 ; n_index < Arch.NrLayers() + 1; n_index++ )
	{
		vec_ret[n_index] = n_sum;
		n_sum += Arch[n_index];
	}

	return vec_ret;

}


template <> WeightMatrixElementImplementation<double,double>::WeightMatrixElementImplementation( Architecture& Arch  ):
//_structure_layer(Arch),
LayeredImplementation(Arch),
_vec_element( InitializeElementVector(Arch) ),
_vec_accumulative_neurons( InitializeAccumulativeVector(Arch) )
{
}


template <> WeightMatrixElementImplementation<double,double>::WeightMatrixElementImplementation( const WeightMatrixElementImplementation<double,double>& rhs ):
//_structure_layer( Architecture( rhs._structure_layer.ArchVec() ) ),
LayeredImplementation(rhs),
_vec_element				( rhs._vec_element ),
_vec_accumulative_neurons	( rhs._vec_accumulative_neurons )
{
}

template <> WeightMatrixElementImplementation<double,double>::~WeightMatrixElementImplementation()
{
}

template <> NetworkImplementation<double,double>*
CanonicalImplementation<double,double>::Clone() const 
{
	return new CanonicalImplementation<double,double>(*this);
}
*/
/*
template <> WeightMatrixElementImplementation<double,double>&
	WeightMatrixElementImplementation<double,double>::operator=( const WeightMatrixElementImplementation<double,double>& rhs)
{
	if ( this == &rhs )
		return *this;

	//TODO: check if the base class handles this correctly
	// call base class copy assigment oprator:
//	_structure_layer = rhs._structure_layer;

	// destroy all elements, pointed to by the element vector
	size_t n_ind;
	for( n_ind = 0; n_ind < _vec_element.size(); n_ind++ )
		delete _vec_element[n_ind];

	// clear the element vector
	_vec_element.clear();

	// loop over all element of the right hand side element vector
	
	LayerElement<double,double>* p_element;
	for ( n_ind =  0; n_ind < rhs._vec_element.size(); n_ind++ )
	{
		// create a new element which is a copy of the rhs element
		p_element = new LayerElement<double,double>( *rhs._vec_element[n_ind] );

		// push back the pointer
		_vec_element.push_back(p_element);
	}

	// copy the accumulative neuron vector:
	_vec_accumulative_neurons = rhs._vec_accumulative_neurons;

	return *this;
}

template<> size_t WeightMatrixElementImplementation<double,double>::WhichElementInVector(NodeId nid) const
{

	// Returns the place in the pointer to Element vector that this NeuronId 
	// corresponds to.
	// Tested:	26-07-1999
	// By:		Marc de Kamps

	// check if the neuron id is reasonable
	assert ( nid._id_value > 0 && nid._id_value <= static_cast<int>(_structure_layer.NumberOfNodes()) );

	// start at this layer

	int n_layer_index = -1;

	// while nid is greater than the first id of this layer, go one layer higher

	while ( static_cast<size_t>(nid._id_value) >= 
		_vec_accumulative_neurons[ ++n_layer_index ] )
		;

	// return the layer

	return static_cast<size_t>(n_layer_index) - 1;

}


template <> size_t WeightMatrixElementImplementation<double,double>::WhichElementToUpdate(NodeId nid) const
{
	// Gives the ElementID that this element can use internally for the updating of
	// the neuron, corresponding to nid.
	// Tested:	29-07-1999
	// By:		Marc de Kamps


	// check if the neuron id is reasonable
	assert ( nid._id_value >0 && nid._id_value <= static_cast<int>(_structure_layer.NumberOfNodes()) );

	// input neurons should not be updated:
	assert ( WhichElementInVector(nid) > 0 );
	
	// A NeuronId corresponds to a given element, but the update request for this
	// must be made to the previous element!

	return (WhichElementInVector(nid) - 1);
}


template <> ElementId WeightMatrixElementImplementation<double,double>::WhichNeuronInElement( NodeId nid) const
{
	// Returns the position inside the element, corresponding to this NeuronId.
	// Tested:	26-07-1999
	// By:		Marc de Kamps

	// check if the neuron id is reasonable
	assert ( nid._id_value > 0 && nid._id_value <= static_cast<int>(_structure_layer.NumberOfNodes()) );

	// see in what element this neuron id is

	size_t n_layer = WhichElementInVector( nid );
 
	// the accumulative value for this neuron is the first neuron id for this layer
	// ( that corresponds to element 1, as element 0 is reserved for the threshold
	// neuron )

	return ElementId( nid._id_value - _vec_accumulative_neurons[n_layer] + 1);

}



template <> ElementId WeightMatrixElementImplementation<double,double>::WhichNeuronToUpdate(NodeId nid) const
{

	// Find within the Element, the correct ElementId to get neuron nid
	// updated.
	// Tested:	29-07-1999
	// By: Marc de Kamps

	// storage element and update element refer to the same ElementId 

	// check if the neuron id is reasonable
	assert ( nid._id_value >0 && nid._id_value <= static_cast<int>(_structure_layer.NumberOfNodes()) );

	// WhichNeuronInElement gives the correct ElementId for the neuron
	// array, which is on the input side and takes into account the 
	// threshold neuron. On the output side, however, there is no threshold
	// neuron

	return ElementId(WhichNeuronInElement(nid).value - 1);
}


template<> double WeightMatrixElementImplementation<double,double>::Update(NodeId Nid) const
{
	// Call upon the the relevant LayerElement to update the neuron,
	// identified by the NeuronId. Note that for a neuron to  be updated
	// a request must be made to the 'predecessor' element, unlike the case
	// where awe simple want to know its value. Hence the use of specific
	// functions to find the Element and ElementID for updating, which are
	// different from the ones used for value retrieval.
	// Tested:	29-07-1999
	// By:		Marc de Kamps

	// find out which layerelement we need
	// update the appropriate neuron in that element and return the result

	return _vec_element[WhichElementToUpdate(Nid)]->Update(WhichNeuronToUpdate(Nid),Nid);	
}


template <> NodeId WeightMatrixElementImplementation<double,double>::WhichNeuronInput( size_t n_layer_index, ElementId eid ) const
{
	// Given an Element ID and an element index, return the NeuronId of the input neuron
	// Tested:	18-11-1999
	// By:		Marc de Kamps

	assert( n_layer_index < _vec_element.size() );
	assert( eid.value >= 0 && 
                eid.value < static_cast<int>(_vec_element[n_layer_index]->NrInput()) );
	if ( eid.value == 0 )
		return NodeId(0);
	else
		return NodeId( _vec_accumulative_neurons[n_layer_index] + eid.value - 1);
}


template <> NodeId WeightMatrixElementImplementation<double,double>::WhichNeuronOutput( size_t n_layer_index, ElementId eid ) const
{
	// Give the NeuronId of the output neuron corresponding to a given element and ElementId
	// Tested:	18-11-1999
	// By:		Marc de Kamps

	assert( n_layer_index < _vec_element.size() );
	assert( eid.value > 0 && 
                eid.value < static_cast<int>(_vec_element[n_layer_index]->NrInput()) );
	return NodeId( _vec_accumulative_neurons[n_layer_index+1] + eid.value - 1);
}

template <> void WeightMatrixElementImplementation<double,double>::Insert( NodeId nid, double val )
{
	// Tested:	29-07-1999
	// By:		Marc de Kamps
	assert( nid._id_value > 0 );
	_vec_element[WhichElementInVector(nid)]->Insert(WhichNeuronInElement(nid),val);
} 


template <> double WeightMatrixElementImplementation<double,double>::Retrieve(NodeId nid) const
{
	if ( nid._id_value == 0 )
		return  THRESHOLD_VALUE;

	assert( nid._id_value > 0 );
	return _vec_element[WhichElementInVector(nid)]->Retrieve(WhichNeuronInElement(nid));
} 

template <> bool WeightMatrixElementImplementation<double,double>::ReadIn( const Pattern& pat )
{
	// Put a pattern into the implementation
	// Tested:	29-07-1999
	// By:		Marc de Kamps

	assert ( pat.size() == _vec_element[0]->NrInput() - 1 );
	Index n_pat_index;

	// Loop over all pattern elements
	for( n_pat_index = 0; n_pat_index < pat.size(); n_pat_index++ )

		// create the correct neuron ID (implicitely)
		// and insert in the implementation

		Insert( NodeId(n_pat_index+1), pat[n_pat_index]  );

	return true;
}

template <> WeightMatrixElementImplementation<double,double>::WeightReference
	WeightMatrixElementImplementation<double,double>::FromIdToWeightReference( NodeId Out, NodeId In ) const
{
	WeightReference wr_ret;

	if ( In._id_value )
	{

	// In is an ordinary neuron
		assert ( WhichElementInVector(Out) == WhichElementInVector(In) + 1 );

		wr_ret._element	= WhichElementInVector(In);
		wr_ret._in	    = WhichNeuronInElement(In);
		wr_ret._out		= WhichNeuronToUpdate(Out);

	}
	else
	{
		// this concerns a weight from the threshold neuron
		wr_ret._element	= WhichElementToUpdate(Out);
		wr_ret._in		= ElementId(0);
		wr_ret._out		= WhichNeuronToUpdate(Out);

	}

	return wr_ret;
	
}

template <> double WeightMatrixElementImplementation<double,double>::GetWeight(NodeId Out, NodeId In) const
{
	WeightReference wr =  FromIdToWeightReference( Out, In );
	return _vec_element[wr._element]->RetrieveWeight(wr._out,wr._in);
}

template <> void WeightMatrixElementImplementation<double,double>::InsertWeight( NodeId Out,
																	   NodeId In,
																	   double val )
{
	WeightReference wr = FromIdToWeightReference( Out, In );
	_vec_element[wr._element]->InsertWeight(wr._out,wr._in,val);
}

template <> Pattern WeightMatrixElementImplementation<double,double>::ReadOut() const
{
	// Tested: 12-10-1999
	assert (_vec_element.size() > 0 );
	Layer nr_last_layer = static_cast<Layer>(_vec_element.size()) - 1;

	NodeId id_last = BeginId( nr_last_layer );

	// The last element has no output neurons, the output layer of the net
	// are the input neurons of this elements.
	size_t nr_output     = _vec_element[nr_last_layer]->NrInput() - 1;

	Pattern pat_ret( nr_output );
	Index n_output_index;

	for ( n_output_index = 0; n_output_index < nr_output; n_output_index++ )
	{
		NodeId nid_out = NodeId( id_last._id_value + n_output_index );
		pat_ret[n_output_index] =  Retrieve(nid_out);
	}

	return pat_ret;
}

ElementId WeightMatrixElementImplementation<double,double>::InputIndex(Index n_iter, Layer n_layer ) const
{
	Number nr_input = _vec_element[n_layer]->NrInput();
	return ElementId( (n_iter - 1)%nr_input);
}

ElementId WeightMatrixElementImplementation<double,double>::OutputIndex(Index n_iter, Layer n_layer ) const
{
	Number nr_input = _vec_element[n_layer]->NrInput();
	return ElementId( (n_iter-1)/nr_input );
}



template <> void WeightMatrixElementImplementation<double,double>::InsertWeight( const_lyrweight_iter_ref lit, double val)
{
	// get the layer number of the iterator' input layer
	// (this is the element vector index )
	size_t n_element = lit.NrLayer();


	// convert the interator order number two the input and output element id's
	size_t nr_iter = lit.NrIter();
	ElementId in  = InputIndex( nr_iter, n_element );
	ElementId out = OutputIndex( nr_iter, n_element );

	// call upon the element to insert the value

	_vec_element[n_element]->InsertWeight(out,in, val);
}


template <> double WeightMatrixElementImplementation<double,double>::GetWeight( const_lyrweight_iter_ref lit) const
{
	// get the layer number of the iterator' input layer
	// (this is the element vector index )
	size_t n_element = lit.NrLayer();


	// convert the interator order number two the input and output element id's
	size_t nr_iter = lit.NrIter();
	ElementId in  = InputIndex( nr_iter, n_element );
	ElementId out = OutputIndex( nr_iter, n_element );

	// call upon the element to insert the value

	return _vec_element[n_element]->RetrieveWeight(out,in);
}

template <> NodeId WeightMatrixElementImplementation<double,double>::NeuronInId( const_lyrweight_iter_ref lit) const
{
	size_t n_element = lit.NrLayer();
	size_t nr_iter = lit.NrIter();

	ElementId in = InputIndex( nr_iter, n_element );
	return WhichNeuronInput( n_element, in );
}


template <> NodeId WeightMatrixElementImplementation<double,double>::NeuronOutId( const_lyrweight_iter_ref lit) const
{
	size_t n_element = lit.NrLayer();
	size_t nr_iter = lit.NrIter();

	ElementId out = OutputIndex( nr_iter, n_element );
	out.value++;
	return WhichNeuronOutput( n_element, out );
}


template <> bool WeightMatrixElementImplementation<double,double>::ToStream( std::ostream& s )
{

	s << "---- Begin Network Implementation ----" << std::endl;
	s << "---- Begin Info ----"			<< std::endl;
	s << "Implementation: Canonical"    << std::endl;

	Index n_layer_index;

	s << "Layer\t\tNumber of relevant neurons\n";

	for ( n_layer_index = 0; n_layer_index < _vec_element.size(); n_layer_index++ )
		s << n_layer_index << "\t\t" << _vec_element[n_layer_index]->NrInput() - 1 << std::endl;

	s << "---- End Info ----"			<< std::endl;
	s << "---- Begin Neuron Values ----"<< std::endl;
	s << "ElementIndex\tElementId\tNeuronId\tvalue"
										<< std::endl;

	Index n_element_index;
	for ( n_layer_index = 0; n_layer_index < _vec_element.size(); n_layer_index++ )
	{
		s << n_layer_index << "\t\t" << 0 << "\t\t" << "threshold" 
		  << "\t\t"        << _vec_element[n_layer_index]->Retrieve(ElementId(0)) 
		  << std::endl;

		for (	n_element_index = 1; 
				n_element_index < _vec_element[n_layer_index]->NrInput(); 
				n_element_index++ )
		{
			NodeId id_which = WhichNeuronInput(n_layer_index,ElementId(n_element_index));
			s << n_layer_index	 << "\t\t" << n_element_index		<< "\t\t" 
			  << id_which._id_value   << "\t\t"
			  << Retrieve( id_which )
			  << std::endl;
		}
	}

	s << "---- End Neuron Values ----"		<< std::endl;
	s << "---- Begin Weight Values ----"	<< std::endl;

	Number n_input;
	Number n_output;

	for ( n_layer_index = 0; n_layer_index < _vec_element.size(); n_layer_index++ )
	{
		for ( n_input = 0; n_input < _vec_element[n_layer_index]->NrInput(); n_input++ )
			for ( n_output = 0; n_output < _vec_element[n_layer_index]->NrOutput(); n_output++ )
				s	<< n_layer_index	
					<< "\t" << n_output 
					<< "\t" << n_input	<< "\t"				
					<< _vec_element[n_layer_index]->RetrieveWeight(ElementId(n_output),ElementId(n_input))
					<< std::endl;
	}

	s << "---- End Weight Values ----"		<< std::endl;
	s << "---- End Network Implementation ----"		<< std::endl;

	return true;
}



size_t WeightMatrixElementImplementation<double,double>::NrNeurons() const
{
	return _structure_layer.NumberOfNodes();
}

size_t WeightMatrixElementImplementation<double,double>::NrInput() const
{
	return _structure_layer.NumberOfInputNodes();
}

size_t WeightMatrixElementImplementation<double,double>::NrOutput() const
{
	return _structure_layer.NumberOfOutputNodes();
}

size_t WeightMatrixElementImplementation<double,double>::NrLayers() const
{
	return _structure_layer.NumberOfLayers();
}

NodeId WeightMatrixElementImplementation<double,double>::BeginId( Layer l ) const
{
	return _structure_layer.BeginId(l);
}

size_t WeightMatrixElementImplementation<double,double>::NrConnections() const
{
	return _structure_layer.NrConnections();
}

size_t WeightMatrixElementImplementation<double,double>::NrLargestLayer() const
{
	return _structure_layer.MaxNumberOfNodesInLayer();
}

size_t WeightMatrixElementImplementation<double,double>::NrNeuronsInLayer( Layer l ) const
{
	return _structure_layer.NumberOfNodesInLayer(l);
}

size_t WeightMatrixElementImplementation<double,double>::NrConnectionFrom( Layer l ) const
{
	return _structure_layer.NrConnectionFrom(l);
}
*/

