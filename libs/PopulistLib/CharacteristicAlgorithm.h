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
#ifndef _CODE_LIBS_POPULISTLIB_CHARACTERISTICALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_CHARACTERISTICALGORITHM_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "CharacteristicParameter.h"

using DynamicLib::AlgorithmGrid;
using DynamicLib::NodeState;
using DynamicLib::SimulationRunParameter;

namespace PopulistLib {

	//! General population density solver algorithmm.

	//! This algorithm implements a population density solver for any arbitrary 1D point model neuron. The
	//! point model neuron is encoded as a CharacteristicSolver object, which may encode either a numerical
	//! or an analytic solution of the ordinary differential equation corresponding for the neuronal point model.
	//! This algorithm then applies the interaction with Poisson distributed input spikes and a population of
	//! point model neurons.

	template <class Weight>
	class CharacteristicAlgorithm : public AbstractAlgorithm<Weight> {
	public:

		typedef typename AbstractAlgorithm<Weight>::predecessor_iterator predecessor_iterator;

		//! 
		CharacteristicAlgorithm(const CharacteristicParameter&);

		//! copy constructor
		CharacteristicAlgorithm(const CharacteristicAlgorithm&);

		//! virtual destructor
		virtual ~CharacteristicAlgorithm();


		//! Copy internal grid, pure virtual
		//! because the representation of a state may determined by the particular algorithm
		virtual AlgorithmGrid Grid() const;

		//! Copy NodeState, in general a much simpler object than the AlgorithmGrid,
		//! from which it is calculated. Usually something simple, like a rate
		virtual NodeState  State() const;

		//! An algorithm saves its log messages, and must be able to produce them for a Report
		virtual string LogString() const;

		//! Cloning operation, to provide each DynamicNode with its own 
		//! Algorithm instance
		virtual CharacteristicAlgorithm<Weight>* Clone() const;

		//! A complete serialization of the state of an Algorithm, so that
		//! it can be resumed from its disk representation. NOT IMPLEMENTED.
		virtual bool Dump(ostream&) const ;

		//! Configure the algorithm with run time parameter
		virtual bool Configure
		(
			const SimulationRunParameter&
		);

		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time
		);

		//! Current internal time as maintained by the algorithm, which may be different from the 'network' time.
		virtual Time CurrentTime() const;

		//! Current firing rate or activation
		virtual Rate CurrentRate() const;

	private:

		Time			_t_current;
		Rate			_rate_current;
		AlgorithmGrid	_grid;
		NodeState		_state;

	};

}

#endif // include guard