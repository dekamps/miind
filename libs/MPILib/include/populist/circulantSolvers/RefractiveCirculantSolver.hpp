// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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
#ifndef MPILIB_POPULIST_CIRCULANTSOLVERS_REFRACTIVECIRCULANTSOLVER_HPP_
#define MPILIB_POPULIST_CIRCULANTSOLVERS_REFRACTIVECIRCULANTSOLVER_HPP_

#include <MPILib/include/populist/circulantSolvers/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/ProbabilityQueue.hpp>

namespace MPILib {
namespace populist {
namespace circulantSolvers {

	//! This AbstractCirculantSolver subclass stores proability density for, presumably LIF, neurons in a priority queue. Immediately
	//! after the refractive period ends for those neurons, the corrsponding prability is reintroduced in the reset bin. 

	//! RefractiveCirculantSolver is designed to run before any NonCirculantSolver. The perecision argument of the
	//! RefractiveCirculantSolver and the NonCirculantSolver should match, otherwise the probability sum between them may differ
	//! from one.

	class RefractiveCirculantSolver : public AbstractCirculantSolver {
	public:

		//! standard constructor
		RefractiveCirculantSolver
		(
			Time t_ref,								//! Refractive period of the neuron
			Time t_batch = 1e-4,					//! Batch size for storing the probability density in the queue, i.e. the precision by which proability density is maintained.
			double precision = 0,					//! Specifying a probability implies the assumption that in the NonCirculantSolver all calculations are broken off after terms such as $\f  \frac{\tau^k}{\tau !}e^{-tau}
			CirculantMode mode = INTEGER			//! Integer only transports probability between bins that are an integer step away
		):AbstractCirculantSolver(mode),_t_ref(t_ref),_t_batch(t_batch),_precision(precision){}

		//! Carry out solver operation
		virtual void 
			Execute
			(
				Number, //!< current number of bins
				Time,	//!< time to solve for
				Time	//!< current simulation time
			);

		//! destructor
		virtual ~RefractiveCirculantSolver(){}

		//! cloning
		virtual RefractiveCirculantSolver* Clone() const {return new RefractiveCirculantSolver(*this); }

		//! return the current refractive time
		Time TimeRefractive() const { return _t_ref; }

		//! return the total probability in the refractive queue
		virtual double RefractiveProbability() const { return _off_queue + _queue.TotalProbability(); }

		//! probability from the refractive queue is retintroduced in the reset bin.
		virtual void AddCirculantToState(Index);

		//! In this solver the NonCirculant must be executed first
		virtual bool BeforeNonCirculant() {return false;}

		//! This circulant need to rescale the probability held in the queue after rebinning.
		virtual void ScaleProbabilityQueue(double scale){_queue.Scale(scale);}


	private:

		double AboveThreshold(Time) const;

		Time				_t_ref;
		Time				_t_batch;
		ProbabilityQueue	_queue;
		double				_precision;
		double				_off_queue = 0.0;
	};
} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_CIRCULANTSOLVERS_REFRACTIVECIRCULANTSOLVER_HPP_

