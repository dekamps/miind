// Copyright (c) 2005 - 2014 Marc de Kamps
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
#ifndef _CODE_LIBS_MPILIB_PRIORITYQUEUE_INCLUDE_GUARD
#define _CODE_LIBS_MPILIB_PRIORITYQUEUE_INCLUDE_GUARD

#include <queue>
#include <MPILib/include/BasicDefinitions.hpp>
#include "StampedProbability.hpp"

using std::queue;

namespace MPILib {
namespace populist {

	//! For a refractive neuron it may be necessary to store probability density. Probability density is held in a ProbabilityQueue
	//! for the time that the neuron is refractive. To be able to do this all probability density corresponding to neurons pushed over
	//! threshold is time stamped, held in a queue and released when external time has moved beyond a period longer than the refractive
	//! period then reintroduced in the reset bin. This reintroduction is typically done by an AbstractCirculantSolver.

	//! For convenience probability is grouped in batches, typically for a period of which TIME_REFRACT_MIN by default. 
	//! Let this period be \f$\t_{batch}\f$. Probability is now time stamped by means of a StampedProbability, which lists the
	//! amount of probability and the time of storage, the event time \f$t_{event}$\f. The queue will maintain an internal
	//! time based on the CollectAndRemove method. Upon calling this method all probability up to \f$ t = n t_{batch}\f$ with
	//! \f$n = \div(t, t_{batch})\f$ will be released and remove from the queue. For example, if \f$t_{batch} =0.1$\f and three
	//! stamped events have been pushed on to the queue, with \f$t = 0.99, 1.0, 1.05$\f, then call CollectandRemove(1.0) will
	//! remove the first two events, but not the last. The internal queue time will also be set to \f$t\f$, i.e. multiples of
	//! \f$t_{batch}$\f. It is illegal to push and event on the queue with a time that is 
	//! earlier than the current time of the queue and this will trigger a PopulistException. It is legal to push events
	//! which are not time ordered on the queue.

	class ProbabilityQueue {
	public:
		//! Probability is grouped in batches
		ProbabilityQueue(Time time_step = TIME_REFRACT_MIN);

		//! destructor
		~ProbabilityQueue(){}

		//! push time stamped probability on the queue
		void push(const StampedProbability& prob);

		//! if there is time stamped probability that would be retrieved by CollectAndRemove before this will return true, false otherwise
		bool HasProbability(Time) const;

		//! add all probability that is batched below the current time and remove it from the queue
		Probability CollectAndRemove(Time);

		//! Total probability in queue
		Probability TotalProbability() const {return _scale*_total;}

		//! Stamped Probability must entered in the queue in the right time order
		bool IsConsistent() const;

		//! Current time based on last CollectAndRemove call
		Time TimeCurrent() const { return _t_current; }

		//! Sometimes, after rebinning the probability in the queue needs to be rescaled
		void Scale(double);

		Time TBatch () const { return _t_batch_size; }

	private:

		double _scale;
		Time _t_batch_size;
		Time _t_current;
		Time _t_current_batch;

		Probability _prob_current_batch;
		Probability _total;

		std::queue<StampedProbability> _queue;

	};
} // populist
} //MPILib
#endif // include guard
