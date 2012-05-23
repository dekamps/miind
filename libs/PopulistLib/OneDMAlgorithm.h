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
#ifndef _CODE_LIBS_POPULISTLIB_ONEDMALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_ONEDMALGORITHM_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "ABConvertor.h"
#include "ABScalarProduct.h"
#include "BasicDefinitions.h"
#include "OneDMParameter.h"
#include "OneDMZeroLeakEquations.h"
#include "OrnsteinUhlenbeckConnection.h"
#include "PopulationGridController.h"
#include "PopulistParameter.h"
#include "OrnsteinUhlenbeckParameter.h"
#include <sstream>

using DynamicLib::AbstractAlgorithm;
using DynamicLib::AlgorithmGrid;
using DynamicLib::DynamicNetwork;
using DynamicLib::DynamicNetworkImplementation;
using DynamicLib::DynamicNode;
using DynamicLib::NodeState;
using DynamicLib::RateAlgorithm;
using DynamicLib::RateFunctor;
using DynamicLib::SimulationRunParameter;

namespace PopulistLib {

	//! This is an implementation for the 1DM Markov process of Muller et al. (2007)
	//! http://dx.doi.org/10.1162/neco.2007.19.11.2958
	//! \deprecated


	template <class Weight>
	class OneDMAlgorithm : public AbstractAlgorithm<Weight> {
	public:

		typedef typename AbstractAlgorithm<Weight>::predecessor_iterator predecessor_iterator;

		//! 
		OneDMAlgorithm(const OneDMParameter&);

		//!
		OneDMAlgorithm(const OneDMAlgorithm&);

		//! 
		virtual ~OneDMAlgorithm();

		//! 
		bool Configure(const SimulationRunParameter&);

		//!
		virtual bool EvolveNodeState
		(
			predecessor_iterator, 
			predecessor_iterator, 
			Time
		);
		//! 
		virtual Time CurrentTime() const;

		//! Give the current output rate
		virtual Rate CurrentRate() const;

		//! Provide a copy of the momentary grid
		virtual AlgorithmGrid Grid() const;

		//! State of the current algorithm exported as a NodeState
		virtual NodeState State() const;

		//! 
		virtual string LogString() const;

		//! Provide a clone of this algorithm
		virtual OneDMAlgorithm<Weight>* Clone() const {return new OneDMAlgorithm<Weight>(*this);}

		//! Dump the current state of the algorithm
		bool Dump(ostream&) const;

		virtual bool Values() const {return false;}

		virtual vector<ReportValue> GetValues() const { return vector<ReportValue>(0); }

	private:

		void WriteConfigurationToLog();

		// conversion for the purpose of setting the controller correctly.
		PopulationParameter							ToPopulationParameter(const OneDMParameter&);

		OneDMParameter								_parameter_onedm;

		mutable ostringstream						_stream_log; // before AlgorithmGrid, which receives a pointer to this stream

		AlgorithmGrid								_grid;

		VALUE_MEMBER

		PopulationGridController<Weight>			_controller_grid;

		Time										_current_time;
		Rate										_current_rate;

	};
}

#endif //include guard