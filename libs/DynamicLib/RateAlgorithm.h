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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_LIBS_DYNAMICLIB_RATEALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_RATEALGORITHM_INCLUDE_GUARD

#include <utility>
#include "AbstractAlgorithmCode.h"
#include "LocalDefinitions.h"

using std::pair;

namespace DynamicLib {

	//! RateAlgorithm
	template <class WeightValue>
	class RateAlgorithm : public AbstractAlgorithm<WeightValue>
	{
	public:

		typedef typename DynamicNode<WeightValue>::predecessor_iterator predecessor_iterator;
		//! 
		RateAlgorithm( istream& s);

		//! RateAlgorithm: construct an algorithm that produces a stationary rate
		RateAlgorithm( Rate );

		//! Construct an algorithm that produces a stationary rate, whose values is maintained by an external variable. The variable
		//! must have a well defined numerical value when a DynamicNetwork using this RateAlgorithm is configured. If this is not the
		//! case, the simulation results are undefined.
		RateAlgorithm( Rate* );

		//! copy constructor
		RateAlgorithm(const RateAlgorithm&);

		//! destructor
		virtual ~RateAlgorithm();

		//! evolution algorithm, which in this case just amounts to returning the current Rate
		virtual bool EvolveNodeState
		(
				predecessor_iterator, 
				predecessor_iterator,
				Time
		);

		//! Copy internal grid
		virtual AlgorithmGrid Grid() const;

		//! Copy NodeState
		virtual NodeState State() const;

		virtual string LogString() const;

		virtual RateAlgorithm* Clone() const;

		virtual bool Dump(ostream&) const;

		virtual bool Configure
		(
				const SimulationRunParameter&
		);

		virtual Time CurrentTime() const;

		virtual Rate CurrentRate() const;

		virtual bool ToStream(ostream&) const;

		virtual bool FromStream(istream&);

		virtual string Tag() const;

	private:

		void	StripFooter			(istream&);

		Time         _time_current;
		Rate         _rate;
		Rate*	     _p_rate;

	}; // end of RateAlgorithm 

	typedef RateAlgorithm<double> D_RateAlgorithm;

} // end of DynamicLib

#endif // include guard
