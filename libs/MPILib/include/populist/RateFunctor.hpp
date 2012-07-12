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

#ifndef MPILIB_POPULIST_RATEFUNCTOR_HPP_
#define MPILIB_POPULIST_RATEFUNCTOR_HPP_

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>

namespace MPILib {
namespace populist {

typedef Rate (*RateFunction)(Time);

inline Rate Nul(Time) {
	return 0;
}

//! An Algorithm that encapsulates a rate as a function of time
//!
//! It is sometime necessary to provide a network with external inputs. These inputs are created as
//! DynamicNode themselves. Their state is trivial and their output firing rate, given by the
//! CurrentRate method follows a given function of time.
template<class WeightValue>
class RateFunctor: public algorithm::AlgorithmInterface<WeightValue> {
public:

	//! Constructor must be initialized with pointer a rate function of time.
	RateFunctor(RateFunction);

	//! mandatory virtual destructor
	virtual ~RateFunctor() {
	}

	//! Essentially just calling the encapsulated rate function. The connection iterators are
	//! ignored and essentially just the current simulation time is set.
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<WeightValue>& weightVector, Time time) override;

	//! Gives the current rate according to the original rate function
	virtual Rate getCurrentRate() const override{
		return _function(_current_time);
	}

	//! Mandatory Grid  function, not of practical use.
	virtual algorithm::AlgorithmGrid getGrid() const override;

	virtual RateFunctor* clone() const override;

	virtual void configure(const SimulationRunParameter&) override;

	//! Gives the current time that the Algorithm keeps.
	virtual Time getCurrentTime() const override;

private:

	RateFunction _function;
	Time _current_time = 0.0;

};
// end of rateFunctor

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_RATEFUNCTOR_HPP_
