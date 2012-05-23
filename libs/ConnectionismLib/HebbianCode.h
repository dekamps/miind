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
#ifndef _CODE_LIBS_CONNECTIONISMLIB_HEBBIANCODE_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISMLIB_HEBBIANCODE_INCLUDE_GUARD

#include <cmath>
#include "Hebbian.h"

namespace ConnectionismLib
{

	template <class Implementation>
	Hebbian<Implementation>::Hebbian
	(
		const HebbianTrainingParameter& par_train
	):
	_p_imp(0),
	_par_train(par_train)
	{
	}

	template <class Implementation>
	Hebbian<Implementation>::Hebbian
	(
		implementation_pointer			p_imp, 
		const HebbianTrainingParameter&	par_train
	):
	_p_imp(p_imp),
	_par_train(par_train)
	{
	}

	template <class Implementation>
	Hebbian<Implementation>::~Hebbian()
	{
	}

	template <class Implementation>
	Hebbian<Implementation>* Hebbian<Implementation>::Clone
		( 
			implementation_pointer			 p_implementation
		) const
	{
		Hebbian<Implementation>* p_training = 
			new Hebbian<Implementation>
				( 
						p_implementation, 
						_par_train
				);
		return p_training;
	}

	template <class Implementation>
	bool Hebbian<Implementation>::Initialize()
	{
		// loop over all layers except the first
		Layer number_of_layers = _p_imp->NumberOfLayers(); 
		for( Layer n_layer = 1; n_layer < number_of_layers; n_layer++ )
		{
			// Note that no threshold weights will be trained:
			typedef typename Implementation::WeightLayerIterator iterator;
			iterator* p_dummy = 0;

			for ( iterator iter_node =  _p_imp->begin(n_layer, p_dummy);
				           iter_node != _p_imp->end  (n_layer, p_dummy);
						   iter_node++
				)
					for
					(
						typename Implementation::NodeType::predecessor_iterator iter_predecessor = iter_node->begin();
						iter_predecessor != iter_node->end();
						iter_predecessor++
					)
						iter_predecessor.SetWeight(0);
		}

		return true;
	}

	template <class Implementation>
	bool Hebbian<Implementation>::Train
	(
		const TrainingUnit<typename Implementation::NodeValue>& tu
	)
	{
		// loop over all layers except the first

		Layer number_of_layers = _p_imp->NumberOfLayers();
		for( Layer n_layer = 1; n_layer < number_of_layers; n_layer++ )
		{
		// Note that no threshold weights will be trained:
			typedef typename Implementation::WeightLayerIterator iterator;
			iterator* p_dummy = 0;

			for ( iterator iter_node =  _p_imp->begin(n_layer, p_dummy);
				           iter_node != _p_imp->end  (n_layer, p_dummy);
						   iter_node++)
				for 
				(
					typename Implementation::NodeType::predecessor_iterator iter_predecessor = iter_node->begin();
					iter_predecessor != iter_node->end();
					iter_predecessor++
				)
				{
					double f_output = iter_node->GetValue();

					NodeId id_input = iter_predecessor->MyNodeId();
	
					if ( id_input != NetLib::THRESHOLD_ID )
					{
						double f_input      = iter_predecessor->GetValue();
						// the fudge factors base and scale are intended for multi-layer feedback networks
						// if the Hebbian training uses small activations, an attenuation factor will arise
						// which amplified with each value. Since this fudge factor addressess all weights in one layer
						// an Hebbian training is determined up to an overall constant, this is reasonable
						double f_weight     = f_output*f_input*_par_train._scale*pow(_par_train._base,static_cast<int>(n_layer));
						double f_old_weight = iter_predecessor.GetWeight();
						iter_predecessor.SetWeight(f_weight+f_old_weight);
					}
				}

			}	

		return true;
	}

} // end of Connectionism

#endif // include guard
