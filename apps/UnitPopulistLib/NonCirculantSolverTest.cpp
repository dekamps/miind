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
#include <valarray>

using namespace PopulistLib;
using namespace std;

namespace {

	class TestSolver : public AbstractNonCirculantSolver {
	public:

		TestSolver(CirculantMode mode):AbstractNonCirculantSolver(mode){}

		TestSolver(const TestSolver& rhs):AbstractNonCirculantSolver(rhs){}

		virtual ~TestSolver(){}

		virtual AbstractNonCirculantSolver* Clone() const { return new TestSolver(*this); }

		virtual void ExecuteExcitatory(Number, Time){}

		virtual void ExecuteInhibitory(Number, Time){}

		void FakeInitializeArrayFactor(Time t, Number n){ this->InitializeArrayFactor(t, n); }
	};
}

BOOST_AUTO_TEST_CASE(AbstractNonCirculantSolverTest){

	TestSolver solver(INTEGER);

	Time t = 0.1;               // t and fraction are the only two numbers that matter in this test
	double fraction = 0.001;	// the calculation of array_rho is broken off after n_terms

	Number number_state = 10;
	InputParameterSet set;
	valarray<double> varray(number_state);
	solver.Configure(varray,set,fraction);
	solver.FakeInitializeArrayFactor(t,number_state);
	Number n_terms = solver.NumberFactor();
	BOOST_REQUIRE(n_terms == 3); // i.e. upto cubic terms must be accounted for

}

BOOST_AUTO_TEST_CASE(IntegralNonCirculantSolverTest){

	const int n_non_circ = 7;
	valarray<double> array_state(n_non_circ); 
	InputParameterSet set;
	set._H_exc = 2; // jump 2 bins
	set._n_noncirc_exc = n_non_circ;

	NonCirculantSolver solver(INTEGER);
	solver.Configure(array_state,set);

	array_state = 0;
	array_state[0] = 1.0;
	const Time tau = 0.1;

	// therefore probility ends up in every second bin
	solver.ExecuteExcitatory(n_non_circ,tau);
	BOOST_REQUIRE(array_state[0] == exp(-tau) );
	BOOST_REQUIRE(array_state[2] == tau*exp(-tau));
	BOOST_REQUIRE(array_state[4] ==  tau*tau*exp(-tau)/2);
}

BOOST_AUTO_TEST_CASE(FractionalNonCircuitSolver){

	// in this case the probility transported from bin 0 should be divided over bin 2 and 3 and with some of it ending in bin 5
	const int n_non_circ = 8;
	valarray<double> array_state(n_non_circ);
	InputParameterSet set;
	set._rate_exc  = 1.0;
	set._H_exc     = 2;
	set._alpha_exc = 0.5;
	set._n_noncirc_exc = n_non_circ;

	NonCirculantSolver solver(FLOATING_POINT);
	solver.Configure(array_state,set);

	array_state = 0.0;
	array_state[0] = 1.0;
	const Time tau = 0.1;
	solver.ExecuteExcitatory(n_non_circ,tau);
	BOOST_REQUIRE(array_state[0] == exp(-tau));
	BOOST_REQUIRE(array_state[2] == 0.5*tau*exp(-tau) );
	BOOST_REQUIRE(array_state[3] == 0.5*tau*exp(-tau) );
	BOOST_REQUIRE(array_state[5] == tau*tau*exp(-tau)/2);

}

BOOST_AUTO_TEST_CASE(TestEpsilon){
	// When any non-circulant solver is used with a finite epsilon, the number of terms calculated in array_rho
	// is limited. The ExecuteExciatory and ExecuteInihibitory algorithms must corrtectly handle this.

	const int n_non_circ = 8;
	valarray<double> array_state(n_non_circ);
	InputParameterSet set;
	set._h_exc = 2.5;
	set._n_noncirc_exc = n_non_circ;
	Time t  = 0.01;
	double accuracy = 0.01;
	TestSolver solver(INTEGER);
	solver.Configure(array_state,set,accuracy);

	// n_non_circ is much more than what is actually required, but because of the limited accuracy
	// it should only calculate n entries
	solver.FakeInitializeArrayFactor(t,n_non_circ);

	int n = solver.NumberFactor(); // this is the maximum number of terms that the solver may take into  account
	BOOST_REQUIRE(n == 1);
}


BOOST_AUTO_TEST_CASE(TestEpsilonLarge){
	// Same as TestEpsilon but for large t.

	const int n_non_circ = 8;
	valarray<double> array_state(n_non_circ);
	InputParameterSet set;
	set._h_exc = 2.5;
	set._n_noncirc_exc = n_non_circ;
	Time t  = 2.5;
	double accuracy = 0.01;
	TestSolver solver(INTEGER);
	solver.Configure(array_state,set,accuracy);

	// n_non_circ is much more than what is actually required, but because of the limited accuracy
	// it should only calculate n entries
	solver.FakeInitializeArrayFactor(t,n_non_circ);

	int n = solver.NumberFactor(); // this is the maximum number of terms that the solver may take into  account
	BOOST_REQUIRE(n == 7);
}

BOOST_AUTO_TEST_CASE(LimitedAccuracyNonCirculant){
	// same test as FractionalNonCircuitSolver but with a deliberately limited accuracy
	const int n_non_circ = 8;
	valarray<double> array_state(n_non_circ);
	InputParameterSet set;
	set._H_exc         = 2;
	set._alpha_exc     = 0.5;
	set._n_noncirc_exc = n_non_circ;
	set._rate_exc      = 1.0;

	double accuracy =  0.01;
	NonCirculantSolver solver(FLOATING_POINT);
	solver.Configure(array_state,set,accuracy);

	array_state[0] = 1.0;
	const Time tau = 0.1;
	solver.ExecuteExcitatory(n_non_circ,tau);
	BOOST_REQUIRE(array_state[0] == exp(-tau));
	BOOST_REQUIRE(array_state[2] == 0.5*tau*exp(-tau) );
	BOOST_REQUIRE(array_state[3] == 0.5*tau*exp(-tau) );
	BOOST_REQUIRE(array_state[5] == 0); // the last term, which is smaller than accuracy is discounted

}

BOOST_AUTO_TEST_CASE(CompareLinearFractional){
	// of course, for integer steps, the fractional and integer algorithm should give the
	// same values

	const int n_non_circ = 7;
	valarray<double> array_state(n_non_circ); 
	InputParameterSet set;
	set._H_exc         = 2; // jump 2 bins
	set._alpha_exc     = 0;
	set._rate_exc      = 1.0;
	set._n_noncirc_exc = n_non_circ;

	NonCirculantSolver solver(INTEGER);
	solver.Configure(array_state,set);

	array_state = 1.0/n_non_circ;
	const Time tau = 0.3;

	// therefore probility ends up in every second bin
	solver.ExecuteExcitatory(n_non_circ,tau);

	NonCirculantSolver fracsolver(FLOATING_POINT);
	valarray<double> array_state_fp(n_non_circ);
	set._h_exc = set._H_exc;
	array_state_fp = 1.0/n_non_circ;
	fracsolver.Configure(array_state_fp,set);

	fracsolver.ExecuteExcitatory(n_non_circ,tau);

	for (int i = 0; i < n_non_circ; i++)
		BOOST_CHECK_CLOSE(array_state[i],array_state_fp[i],1e-9);

}