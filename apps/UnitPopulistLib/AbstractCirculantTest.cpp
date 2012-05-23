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
#include <boost/test/execution_monitor.hpp>
#include <PopulistLib.h>

using namespace PopulistLib;

namespace {

	class TestCirculantSolver : public AbstractCirculantSolver {
	public:

		TestCirculantSolver(CirculantMode mode):AbstractCirculantSolver(mode){}

		virtual ~TestCirculantSolver(){}

		virtual TestCirculantSolver* Clone() const { return new TestCirculantSolver(*this); }

		virtual void Execute(Number, Time, Time){}

		void FillNonCirculantBins(){ this->AbstractCirculantSolver::FillNonCirculantBins();}

		double f(int i){return this->_array_rho[i]; }
	};
}

BOOST_AUTO_TEST_CASE( LinearAbstractCirculantSolverTest ) {

	TestCirculantSolver test(INTEGER);
	const int nr_state = 9;
	valarray<double> ar(nr_state);
	ar = 1;
	InputParameterSet par;

	par._H_exc = 3;
	par._n_noncirc_exc= 3;

	test.Configure(&ar,par);
	test.FillNonCirculantBins();

	double val_expected = nr_state/par._H_exc;
	for (Index i = 0; i < par._n_noncirc_exc; i++)
		BOOST_CHECK(val_expected == test.f(i));
}

BOOST_AUTO_TEST_CASE( FPAbstractCirculantSolvertest ){

	TestCirculantSolver test(FLOATING_POINT);
	const int nr_state = 10;
	valarray<double> ar(nr_state);
	ar = 1;
	InputParameterSet par;

	par._H_exc         = 2;
	par._alpha_exc     = 0.1;
	par._n_noncirc_exc = 5;

	test.Configure(&ar,par);
	test.FillNonCirculantBins();

	double sum = 0.0;
	for (Index i = 0; i < par._n_noncirc_exc; i++ ){
		if (i < par._n_noncirc_exc - 1)
			BOOST_CHECK_CLOSE(test.f(i), par._H_exc + par._alpha_exc, 1e-9);
		sum += test.f(i);
	}
	BOOST_CHECK_CLOSE(sum, nr_state,1e-9);
}

BOOST_AUTO_TEST_CASE( FPACSNiceFit) {
	// nr non_circlant areas fit number of bins exactly
	TestCirculantSolver test(FLOATING_POINT);
	const int nr_state = 10;
	valarray<double> ar(nr_state);
	ar = 1;
	InputParameterSet par;

	par._H_exc         = 2;
	par._alpha_exc     = 0.5;
	par._n_noncirc_exc = 4;

	test.Configure(&ar,par);
	test.FillNonCirculantBins();

	double sum = 0.0;
	for (Index i = 0; i < par._n_noncirc_exc; i++ ){
		BOOST_CHECK_CLOSE(test.f(i), par._H_exc + par._alpha_exc, 1e-9);
		sum += test.f(i);
	}
	BOOST_CHECK_CLOSE(sum, nr_state,1e-9);
}

BOOST_AUTO_TEST_CASE( FPACSIntegerFit) {
	// jump is actually integer, but nr non circulant areas does not fit
	TestCirculantSolver test(FLOATING_POINT);
	const int nr_state = 9;
	valarray<double> ar(nr_state);
	ar = 1;
	InputParameterSet par;

	par._H_exc         = 2;
	par._alpha_exc     = 0.0;
	par._n_noncirc_exc = 5;

	test.Configure(&ar,par);
	test.FillNonCirculantBins();

	double sum = 0.0;
	for (Index i = 0; i < par._n_noncirc_exc; i++ ){
		if (i < par._n_noncirc_exc -1 )
			BOOST_CHECK_CLOSE(test.f(i), par._H_exc + par._alpha_exc, 1e-9);
		sum += test.f(i);
	}
	BOOST_CHECK_CLOSE(sum, nr_state,1e-9);
}

BOOST_AUTO_TEST_CASE( FPACSPerfectFit ){
	// jump is integer, number of non circulant areas fits exacly
	TestCirculantSolver test(FLOATING_POINT);
	const int nr_state = 10;
	valarray<double> ar(nr_state);
	ar = 1;
	InputParameterSet par;

	par._H_exc         = 2;
	par._alpha_exc     = 0.0;
	par._n_noncirc_exc = 5;

	test.Configure(&ar,par);
	test.FillNonCirculantBins();

	double sum = 0.0;
	for (Index i = 0; i < par._n_noncirc_exc; i++ ){
		BOOST_CHECK_CLOSE(test.f(i), par._H_exc + par._alpha_exc, 1e-9);
		sum += test.f(i);
	}
	BOOST_CHECK_CLOSE(sum, nr_state,1e-9);
}

class AddTester : public AbstractCirculantSolver {
public:

	AddTester(CirculantMode mode):AbstractCirculantSolver(mode){}

	void Execute(Number, Time, Time){}

	virtual AddTester* Clone() const{ return new AddTester(*this); }

	virtual ~AddTester(){}

	virtual bool 
		Configure
		(
			valarray<double>* p_state, 
			const InputParameterSet& set
		){
			for (Index i = 0; i < set._n_circ_exc; i++)_array_circulant[i] = 1.0; 
			return AbstractCirculantSolver::Configure(p_state,set); 
		}

private:

};

BOOST_AUTO_TEST_CASE(AddCirculantToStateFP)
{
	Number nr_state = 20;
	AddTester test(FLOATING_POINT);	
	valarray<double> state(nr_state);

	InputParameterSet set;
	set._H_exc      = 2;
	set._alpha_exc  = 0.5;
	set._n_circ_exc =  static_cast<Number>(nr_state/(set._H_exc + set._alpha_exc) + 1);

	test.Configure(&state,set);

	test.AddCirculantToState(0);
	double step = set._H_exc + set._alpha_exc;
	for (Index j = 0; j < nr_state; j++ ){
		if (j/step - floor(j/step) == 0 && j/step < set._n_circ_exc)
			BOOST_CHECK_CLOSE(state[j], 1.0, 1e-10);
		else {
			Index k = static_cast<Index>(j - floor(j/step)*step);
			if ( k == 0){
				double small_frac = j - step*floor(j/step);
				BOOST_CHECK_CLOSE(state[j-1], small_frac, 1e-10);
				BOOST_CHECK_CLOSE(state[j],1-small_frac,1e-10);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(AddCirculantToStateFPUnEqual){
	Number nr_state = 20;
	AddTester test(FLOATING_POINT);	
	valarray<double> state(nr_state);

	InputParameterSet set;
	set._H_exc      = 2;
	set._alpha_exc  = 0.4;
	set._n_circ_exc =  static_cast<Number>(nr_state/(set._H_exc + set._alpha_exc) + 1);

	test.Configure(&state,set);

	test.AddCirculantToState(0);

	double step = set._H_exc + set._alpha_exc;
	for (Index j = 0; j < nr_state; j++ ){
		if (j/step - floor(j/step) == 0 && j/step < set._n_circ_exc)
			BOOST_CHECK_CLOSE(state[j], 1.0, 1e-10);
		else {
			Index k = static_cast<Index>(j - floor(j/step)*step);
			if ( k == 0){
				double small_frac = j - step*floor(j/step);
				BOOST_CHECK_CLOSE(state[j-1], small_frac, 1e-10);
				BOOST_CHECK_CLOSE(state[j],1-small_frac,1e-10);
			}
		}
	}
}


