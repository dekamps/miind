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
#ifndef _CODE_LIBS_CONNECTIONSIMLIB_BACKPROPCODE_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONSIMLIB_BACKPROPCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "BackpropTraining.h"

namespace ConnectionismLib {

	template<class LayeredImplementation, class Iterator>
	vector<Iterator> BackpropTraining<LayeredImplementation, Iterator>::InitializeBeginIterators
	(
		 LayeredImplementation*   p_implementation
	 ) const 
	{
    
		Layer number_of_layers = p_implementation->NumberOfLayers();

		vector<Iterator> vector_return;
		for( Layer n_layer = 0; n_layer < number_of_layers; n_layer++) 
		{
			Iterator* p_dummy = 0;
			Iterator iter = p_implementation->begin(n_layer, p_dummy);

			vector_return.push_back(iter);
		}	
		return vector_return;
	}

  template<class LayeredImplementation, class Iterator>
    vector<Iterator> BackpropTraining<LayeredImplementation, Iterator>::InitializeEndIterators
    (
     LayeredImplementation*   p_implementation
     ) const {

    Layer number_of_layers = p_implementation->NumberOfLayers();

    vector<Iterator> vector_return;
    for( Layer n_layer = 0; n_layer < number_of_layers; n_layer++) {

      Iterator* p_dummy = 0;
      Iterator iter = p_implementation->end(n_layer, p_dummy);

      vector_return.push_back(iter);
    }	
    return vector_return;
  }

	template <class LayeredImplementation, class Iterator>
		BackpropTraining<LayeredImplementation, Iterator>::BackpropTraining
	(
		implementation_pointer		p_implementation,
		const TrainingParameter&	par_train
	):
    _p_implementation			(p_implementation),
     _parameter_train			(par_train),
    _gauss_distribution			(NumtoolsLib::GLOBAL_RANDOM_GENERATOR),
    _vector_of_iterators		(InitializeBeginIterators(p_implementation)),
    _vector_of_begin_iterators	(InitializeBeginIterators(p_implementation)),
    _vector_of_end_iterators	(InitializeEndIterators  (p_implementation)),
    _vector_nodes_buffer		(p_implementation->NumberOfNodes()),
    _vector_weights_buffer		(0) 
	{

		// insert a reverse implementation
		_p_implementation->InsertReverseImplementation();
    
		// see if the iterator is consistent with the desired training
		if (par_train._train_threshold){

			if ( _vector_of_begin_iterators[0]->MyNodeId() != NetLib::THRESHOLD_ID )
				throw InconsistentIteratorException();
		}
	}

  template <class Implementation, class Iterator>
    BackpropTraining<Implementation, Iterator>::BackpropTraining( const TrainingParameter& par_train ):
    _p_implementation(0),
     _parameter_train(par_train),
    _gauss_distribution(NumtoolsLib::GLOBAL_RANDOM_GENERATOR) {

  }

	template <class LayeredImplementation, class Iterator>
	void BackpropTraining<LayeredImplementation, Iterator>::NodesToNodeBuffer()	
	{
		Number number_of_neurons = _p_implementation->NumberOfNodes();
		for (Index index = 0; index < number_of_neurons; index++)
			_vector_nodes_buffer[index] = _p_implementation->Retrieve(NodeId(index));
	}

  template <class LayeredImplementation, class Iterator>
    BackpropTraining<LayeredImplementation, Iterator>*
	    BackpropTraining<LayeredImplementation,Iterator>::Clone
    ( 
		 LayeredImplementation*		p_implementation 
    ) const 
	{ 
		BackpropTraining<LayeredImplementation,Iterator>* p_training =
		 new BackpropTraining<LayeredImplementation,Iterator>
			( 
				p_implementation, 
				_parameter_train
			  );

    return p_training; 
  }

  template <class LayeredImplementation, class Iterator>
    BackpropTraining<LayeredImplementation, Iterator>::~BackpropTraining() {
  }

	template <class Implementation, class Iterator>
	bool BackpropTraining<Implementation, Iterator>::Evolve() 
	{

		for
		( 
			typename Implementation::Order iter = _p_implementation->begin() + _p_implementation->NumberOfInputNodes();
			iter != _p_implementation->end();
			iter++
		)
			iter->Update();
	
		return true;
	}

  template <class Implementation, class Iterator>
    void BackpropTraining<Implementation, Iterator>::CalculateGradient(const D_Pattern& pat_out) {

    // The delta's will replace the Network activities, but we still need them to calculate the
    // new weights, so store them 
    NodesToNodeBuffer();

    // Calculate the delta's at the top most layer
    CalculateInitialGradient(pat_out);

    // Now backpropgate the delta's
    
    // delta's for the output layer already were calculated

    Layer n_output_layer = _p_implementation->NumberOfLayers() - 2;
    for ( Layer n_layer = n_output_layer; n_layer >= 1; n_layer--) {
      for ( 
	   _vector_of_iterators[n_layer] =  _vector_of_begin_iterators[n_layer];
	   _vector_of_iterators[n_layer] != _vector_of_end_iterators[n_layer];
	   _vector_of_iterators[n_layer]++ 
	   ) {

		NodeId id = _vector_of_iterators[n_layer]->MyNodeId();
		if ( id != NetLib::THRESHOLD_ID )
		  {
			  AbstractSquashingFunction& function = _vector_of_iterators[n_layer]->SquashingFunction();
			  double f_inner_product = _vector_of_iterators[n_layer]->InnerProduct();
			  double f_activation = function           ( f_inner_product );
		      double f_derivative = function.Derivative( f_activation    );

			  double f_delta      = f_derivative*(_vector_of_iterators[n_layer].ReverseInnerProduct());
			  _vector_of_iterators[n_layer]->SetValue(f_delta);
		  }
      }
    }
  }

  template <class Implementation, class Iterator>
    void BackpropTraining<Implementation, Iterator>::CalculateInitialGradient(const D_Pattern& pattern_target) {

    Index index_of_output_layer = _p_implementation->NumberOfLayers() - 1;

    Index index_pattern = 0;

    for 
      (
       _vector_of_iterators[index_of_output_layer] =   _vector_of_begin_iterators[index_of_output_layer];
       _vector_of_iterators[index_of_output_layer] !=  _vector_of_end_iterators[index_of_output_layer];
       _vector_of_iterators[index_of_output_layer]++
       )

      // Loop over output layer, calculate delta and store in the output layer
      // of the network
      {
		// Calculate the delta for this Node 
		NodeId id_output_node    = _vector_of_iterators[index_of_output_layer]->MyNodeId();

		if ( id_output_node != NetLib::THRESHOLD_ID ) {
		
			AbstractSquashingFunction& function = _vector_of_iterators[index_of_output_layer]->SquashingFunction();
			double f_output_activity = _vector_of_iterators[index_of_output_layer]->GetValue();
			double f_delta           = f_output_activity - pattern_target[index_pattern++];
			double f_derivative      = function.Derivative
								    ( 
										f_output_activity 
									 );

			_vector_of_iterators[index_of_output_layer]->SetValue( f_derivative*f_delta );

		}
      }
  }

	template <class Implementation, class Iterator>
	bool BackpropTraining<Implementation, Iterator>::Train( const TrainingUnit<typename Implementation::NodeValue>& tu ) 
	{

		// apply the input pattern to the network
		_p_implementation->ReadIn(tu.InPat());
		Evolve(); 

		CalculateGradient(tu.OutPat());
		ApplyGradientToWeights();

		return true;
	}

  template <class LayeredImplementation, class Iterator>
  bool BackpropTraining<LayeredImplementation, Iterator>::Initialize() 
  {
    int number_weights = 0;
    Layer number_of_layers = _p_implementation->NumberOfLayers();
    // loop over all layers
    for (Layer number_layer = 0; number_layer < number_of_layers; number_layer++) 
	{

      // for each layer, use the Iterator to loop over all Nodes
		for 
		(
		 _vector_of_iterators[number_layer] =  _vector_of_begin_iterators[number_layer];
		 _vector_of_iterators[number_layer] != _vector_of_end_iterators[number_layer];
		 _vector_of_iterators[number_layer]++ 
		 )
		{
			NodeId id = _vector_of_iterators[number_layer]->MyNodeId();
			for
			(
				typename LayeredImplementation::NodeType::predecessor_iterator iter_predecessor 
					= _vector_of_iterators[number_layer]->begin();
				iter_predecessor != _vector_of_iterators[number_layer]->end();
				iter_predecessor++
			)
			{
				 if 
				 (
					 ! _parameter_train._train_threshold && 
					 iter_predecessor->MyNodeId() == NetLib::THRESHOLD_ID 
				 )
				 {
						
				      iter_predecessor.SetWeight
				      (
				         _parameter_train._f_threshold_default
				       );
				      number_weights++;
				 }
			
				 else
				 {
					double f_weight = _parameter_train._f_sigma*\
										  _gauss_distribution.NextSampleValue() +
										  _parameter_train._f_bias;

					iter_predecessor.SetWeight
					(
					      f_weight
					);
					number_weights++;
				}
				 
			}
		}
	}

    if ( _parameter_train._f_momentum != 0 )
      _vector_weights_buffer = vector<double>(number_weights);
	return true;
  }


  template <class LayeredImplementation, class Iterator >
  void BackpropTraining<LayeredImplementation, Iterator>::WeightsToWeightBuffer() 
  {
      typedef NetLib::NodeIterator<typename LayeredImplementation::NodeType> Nodeiterator;

      Index index_buffer = 0;
      for
      (
           Nodeiterator   iter = _p_implementation->begin();
	   iter != _p_implementation->end();
	   iter++
      )
      {
	   typedef typename LayeredImplementation::NodeType::predecessor_iterator Weightiterator;
	   for 
	   (
	          Weightiterator witer = iter->begin();
		  witer != iter->end();
		  witer++
	    )
	     _vector_weights_buffer[index_buffer++] = witer.GetWeight();
      }
  }


  template <class LayeredImplementation, class Iterator >
  void BackpropTraining<LayeredImplementation, Iterator>::WeightBufferToWeights() 
  {
      typedef NetLib::NodeIterator<typename LayeredImplementation::NodeType> Nodeiterator;

      Index index_buffer = 0;
      for
      (
           Nodeiterator   iter = _p_implementation->begin();
	   iter != _p_implementation->end();
	   iter++
      )
      {
	   typedef typename LayeredImplementation::NodeType::predecessor_iterator Weightiterator;
	   for 
	   (
	          Weightiterator witer = iter->begin();
		  witer != iter->end();
		  witer++
	    ){
	          witer.SetWeight(_parameter_train._f_momentum*_vector_weights_buffer[index_buffer++] + witer.GetWeight());
	   }
      }
  }
  


  template <class LayeredImplementation, class Iterator>
  void BackpropTraining<LayeredImplementation, Iterator>::ApplyGradientToWeights() {
    if ( _parameter_train._f_momentum != 0 )
      WeightsToWeightBuffer();

    // Updating of the weights starts with a delta and is then applied to all inputs
    // of this node and is then applied to all inputs of the delta node. We therefore
    // start at layer 1

    Layer number_of_layers = _p_implementation->NumberOfLayers();
    for ( Layer n_layer = 1; n_layer < number_of_layers; n_layer++ ) {

		for 
		(
			_vector_of_iterators[n_layer] =  _vector_of_begin_iterators[n_layer];
			_vector_of_iterators[n_layer] != _vector_of_end_iterators[n_layer];
			_vector_of_iterators[n_layer]++
		) 
		{
			for
			(
				typename LayeredImplementation::NodeType::predecessor_iterator iter_predecessor = _vector_of_iterators[n_layer]->begin();
				iter_predecessor !=  _vector_of_iterators[n_layer]->end();
				iter_predecessor++
			)
			{			 
				// Is training actually desired ?
				bool b_train = _parameter_train._train_threshold || 
				iter_predecessor->MyNodeId() != NetLib::THRESHOLD_ID;

			  if ( b_train )
			  {
					double f_activity = _vector_nodes_buffer[iter_predecessor->MyNodeId()._id_value];


				// Get delta
				double f_delta = _vector_of_iterators[n_layer]->GetValue();
	    
			    // Adapt weight
				double f_delta_weight = -_parameter_train._f_stepsize*f_activity*f_delta;
				double f_new_weight = iter_predecessor.GetWeight() + f_delta_weight;
				iter_predecessor.SetWeight(f_new_weight);
			  }

			}
		}
   }

   if ( _parameter_train._f_momentum != 0 )
      WeightBufferToWeights();

   }


} // end of Connectionism


#endif // include guard
