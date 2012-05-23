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
#ifndef _CODE_LIBS_CONNECTIONISM_BACKPROPTRAINING_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISM_BACKPROPTRAINING_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <vector>
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../NetLib/NetLib.h"
#include "TrainingAlgorithm.h"
#include "LayeredNetwork.h"
#include "InconsistentIteratorException.h"

using NumtoolsLib::GaussianDistribution;

namespace ConnectionismLib
{
	//! BackpropTraining

	template <class LayeredImplementation, class Iterator>
	class BackpropTraining  : public TrainingAlgorithm<LayeredImplementation>
	{
	public:

		typedef LayeredImplementation*						implementation_pointer;
		typedef TrainingAlgorithm<LayeredImplementation>	training;

		typedef TrainingUnit<typename LayeredImplementation::NodeValue> TU;  

		//!
		BackpropTraining
			(
				const TrainingParameter&
			);

		//! typical use: give a TrainingParameter and train the network that
		//! is defined by the implementation

		BackpropTraining
			(
				LayeredImplementation*, 
				const TrainingParameter& 
			);
		virtual	~BackpropTraining();


		//! Initialize accoring to a distribution specified by the
		//! TrainingParameter (bias = mu, variance = sigma)

		virtual bool	Initialize();		

		//! perform a single training step
		virtual bool	Train(const TU&);   

		virtual BackpropTraining* Clone
				(
					LayeredImplementation* 
				) const;

	private:


		// auxilliary functions

		vector<Iterator> InitializeBeginIterators(LayeredImplementation*) const;
		vector<Iterator> InitializeEndIterators  (LayeredImplementation*) const;

		void CalculateGradient          (const D_Pattern&);
		void CalculateInitialGradient	(const D_Pattern&);
		void ApplyGradientToWeights     ();
		bool Evolve                     ();

		void NodesToNodeBuffer();
		void NodeBufferToNodes();
		void WeightsToWeightBuffer();
		void WeightBufferToWeights();

		LayeredImplementation*				_p_implementation;
		TrainingParameter					_parameter_train;

		//!Gaussian distribution for initialization
		GaussianDistribution   _gauss_distribution;

		vector<Iterator>		_vector_of_iterators;
		vector<Iterator>        _vector_of_begin_iterators;
		vector<Iterator>        _vector_of_end_iterators;

		vector<double>			_vector_nodes_buffer;
		vector<double>          _vector_weights_buffer;


	}; // end of BackpropTrainingVector

} // end of Connectionism

#endif // include guard
