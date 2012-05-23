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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <PopulistLib.h>

InputParameterSet TestSet(Number state_size) {
	InputParameterSet set;
	set._H_exc         = 2;
	set._alpha_exc     = 0.0;
	set._rate_exc      = 1.0;
	set._n_circ_exc    = static_cast<Number>(state_size/(set._H_exc + set._alpha_exc) + 1);
	set._n_noncirc_exc = static_cast<Number>(state_size/(set._H_exc + set._alpha_exc));

	return set;
}

Time T_SIM = 0;

void SimulateForSomeTime
(
	valarray<double>* p_state, 
	Time tau, 
	RefractiveCirculantSolver* p_sv_c, 
	AbstractNonCirculantSolver* p_sv_nc 
)
{
	T_SIM += tau;
	valarray<double>& state				= *p_state;
	RefractiveCirculantSolver& sv_c		= *p_sv_c;
	AbstractNonCirculantSolver& sv_nc	= *p_sv_nc;	

	sv_nc.ExecuteExcitatory(state.size(), tau);
	sv_c.Execute(state.size(), tau,T_SIM);
	sv_c.AddCirculantToState(0);
}

BOOST_AUTO_TEST_CASE(RefractiveCirculantTest){
	// simulate for less than the refractive period. All circulant stuff should be on queue.
	const Number n_state = 20;
	const double prob_initial = 1.0;
	valarray<double> state(n_state);
	state = 0;
	state[1] = prob_initial;

	Time t_refract		= 4e-3;
	Time t_batch		= 1e-4;
	double precision	= 0.0;

	NonCirculantSolver	sv_nc(FLOATING_POINT);
	RefractiveCirculantSolver sv_c (t_refract,t_batch,precision,FLOATING_POINT);

	InputParameterSet set = TestSet(state.size());

	sv_nc.Configure(state,set);
	sv_c.Configure(&state,set);

	Time t_sim = 0.5e-4;
	SimulateForSomeTime(&state,t_sim,&sv_c,&sv_nc);

	// whatever is lost from the state should be on the queue,
	BOOST_CHECK_CLOSE(prob_initial - state.sum(), sv_c.RefractiveProbability(), 1e-9);
}

BOOST_AUTO_TEST_CASE(LongerThanRefractive){

	const Number n_simulation = 10;
	const Number n_state      = 20;
	const double prob_initial = 1.0;
	valarray<double> state(n_state);
	state = 0;
	state[1] = prob_initial;

	Time t_refract		= 4e-3;
	Time t_batch		= 1e-3;
	double precision	= 0.0;

	NonCirculantSolver	sv_nc(FLOATING_POINT);
	RefractiveCirculantSolver sv_c (t_refract,t_batch,precision,FLOATING_POINT);

	InputParameterSet set = TestSet(state.size());

	sv_nc.Configure(state,set);
	sv_c.Configure(&state,set);

	Time t_sim = 0.5e-3;
	double pmis = 0;
	for (Index i = 0; i < n_simulation; i++)
	{
		SimulateForSomeTime(&state,t_sim,&sv_c,&sv_nc);
		if (i < 1)
			pmis += (1 - state.sum() );
	}
	// after 9 simulations for these parameters, the probability that went above threshold
	// in the first simulation should reappear in the reset bin
	BOOST_CHECK_EQUAL(pmis, state[0]);
	BOOST_CHECK_EQUAL(1.0, state.sum() + sv_c.RefractiveProbability());
}


BOOST_AUTO_TEST_CASE(ZeroRefraction){
}