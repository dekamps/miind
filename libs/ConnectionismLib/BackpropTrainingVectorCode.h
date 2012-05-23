// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef _CODE_LIBS_CONNECTIONISM_BACKPROPTRAININGVECTORCODE_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISM_BACKPROPTRAININGVECTORCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "BackpropTrainingCode.h"
#include "BackpropTrainingVector.h"

namespace ConnectionismLib {

	template <class LayeredImplementation, class Iterator>
    bool BackpropTrainingVector<LayeredImplementation, Iterator>::AcceptTrainingUnitVector
    (
		const vector<TrainingUnit<typename LayeredImplementation::NodeValue> >& vector_training_units
    )
	{

		if ( _vec_tu.size() != 0 )
			return false;
    
	    _vec_tu = vector_training_units;
		_lock = true;
   
    return true;
  }

  template <class LayeredImplementation, class Iterator>
    bool BackpropTrainingVector<LayeredImplementation, Iterator>::CreateCategorization
    (
     const vector<D_Pattern*>& vec_p_pat
     ) {

    // check that no training units have been added on before

    if ( _vec_tu.size() > 0 ){

      // if so: lock the TrainingVector and return false
	
      _lock = true;
      return false;
    }

    // check that the number of input patterns is equal to the number of output neurons

    if ( _p_net->NumberOfOutputNodes() != vec_p_pat.size() ) {
      // if not: lock and  return false
		
      _lock = true;
      return false;
    }

    // loop over the input patterns
    for (Index n_pat_ind = 0; n_pat_ind < vec_p_pat.size(); n_pat_ind++ ){

      // create the corresponding output vector
      D_Pattern Out( vec_p_pat.size() );
      Out.Clear();
      Out[n_pat_ind] = 1;

      // add it to the TUvector
      _vec_tu.push_back( TrainingUnit<typename LayeredImplementation::NodeValue>(*(vec_p_pat[n_pat_ind]), Out));
      
    }
    // lock the vector
    
    _lock = true;

    // return true

    return true;
  }

  template <class LayeredImplementation, class Iterator>
    void BackpropTrainingVector<LayeredImplementation, Iterator>::Train() {
      if ( _mode == BATCH ) {
	}
      else {

	// create a random generator, that is to pick one of the patterns in the vector
	assert ( _mode == RANDOM );
	UniformDistribution ran_uni(NumtoolsLib::GLOBAL_RANDOM_GENERATOR);	

	
	// train  the number of patterns that is in the vector
	// (i.e. the number of trains is equal to those in batch mode, although
	// the patterns are randomly chose)

	for ( Index n_pattern = 0; n_pattern < _vec_tu.size(); n_pattern++ ) {

	  // determine which pattern is to be trained
	  size_t n_which = static_cast<size_t>
                           (
			    ran_uni.NextSampleValue()*_vec_tu.size()
			    );

	  // train the network
	  TrainingUnit<typename LayeredImplementation::NodeValue> pat_pair
	    (
	     _vec_tu[n_which].InPat(), 
	     _vec_tu[n_which].OutPat() 
	     );
	  _p_net->Train( pat_pair );
	} 
      }
    }

  template <class LayeredImplementation, class Iterator>
   double BackpropTrainingVector<LayeredImplementation, Iterator>::ErrorValue
    (
     const D_Pattern& target, 
     const D_Pattern& output
     ) const {
 
   assert ( target.Size() == output.Size() );

   double difsq = 0;

   for (Index n_ind = 0; n_ind < target.Size(); n_ind++ ) {
     difsq += ( target[n_ind] - output[n_ind] )*
       ( target[n_ind] - output[n_ind] );
   }

   return (difsq/target.Size());

  }

  template <class Implementation, class Iterator>
    double BackpropTrainingVector<Implementation, Iterator>::ErrorValue() const  {

    // loop over all training units

    D_Pattern pat_out(_p_net->NumberOfOutputNodes());

    double energy = 0;

    for ( Index nr_tus = 0; nr_tus < _vec_tu.size(); nr_tus++ )	{

      // offer input pattern to the network
		
      _p_net->ReadIn( _vec_tu[nr_tus].InPat() );

      // evolve the network

      _p_net->Evolve();

      // get the output pattern

      pat_out = _p_net->ReadOut();

      // calculate energy for this pattern and add to the total

      energy += ErrorValue( pat_out, _vec_tu[nr_tus].OutPat() );

    }

    // return the calculated energy

    return energy;
  }


	template <class LayeredImplementation, class Iterator>
    BackpropTrainingVector<LayeredImplementation, Iterator>::BackpropTrainingVector
    ( 
		LayeredNetwork<LayeredImplementation>*	p_net, 
		const TrainingParameter&		        par_train,
		TrainingMode							mode 
     ):
    _parameter_train(par_train),
    _mode(mode),
    _lock(false),
    _p_net(p_net) 
	{
		BackpropTraining<LayeredImplementation, Iterator> train(par_train);
		_p_net->SetTraining(train);
    
		for ( Index n_init = 0; n_init < par_train._n_init; n_init++ )
			 _p_net->Initialize();
  }


  template <class LayeredImplementation, class Iterator>
    BackpropTrainingVector<LayeredImplementation, Iterator>::~BackpropTrainingVector() {
  }

template <class LayeredImplementation, class Iterator>
bool BackpropTrainingVector<LayeredImplementation, Iterator>::PushBack
(
  const TrainingUnit<typename LayeredImplementation::NodeValue>& tu 
)
{
    if ( _lock  )
      return false;

    _vec_tu.push_back(tu);
    return true;
  }

} // end of Connectionism

#endif // include guard
