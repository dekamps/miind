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

#include "RefractiveCirculantSolver.h"

using namespace PopulistLib;



double RefractiveCirculantSolver::AboveThreshold(Time tau) const 
{
	double sum = 0.0;
	Number n =  NumberOfSolverTerms(tau, _precision, _p_set->_n_noncirc_exc);
	for (Index i = 0; i < n; i++)
		sum += _array_rho[i];
	return _initial_integral - sum  - _queue.TotalProbability();
}

void RefractiveCirculantSolver::Execute(Number n_bins, Time tau, Time t_sim)
{
	_n_bins = n_bins;
	_tau	= tau;
	// tau is the time that the simulation has to run for
	this->FillNonCirculantBins();
	StampedProbability prob;
	prob._prob = this->AboveThreshold(tau);
	// Most Solvers just run for time tau,
	// but this Solver needs to keep track of total simulation time to make sense of queue unpacking
	prob._time = _t_ref + t_sim; 
	_queue.push(prob);

	_off_queue = _queue.CollectAndRemove(t_sim);
}

void RefractiveCirculantSolver::AddCirculantToState(Index i_reset)
{
	 (*_p_array_state)[i_reset] += _off_queue;
	 _off_queue = 0;  // remember that this value would otherwise influence queries about the total probility in the queue.
}