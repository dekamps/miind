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

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/NetworkState.hpp>
#undef protected
#undef private

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor(){

	NetworkState s = NetworkState(2.0);

	BOOST_CHECK(s._isConfigured==false);
	BOOST_CHECK(s._result==NOT_RUN);
	BOOST_CHECK(s._currentTime==2.0);

}

void test_enum(){
	BOOST_CHECK(static_cast<int>(NOT_RUN) == 0);
	BOOST_CHECK(static_cast<int>(SUCCESS) == 1);
	BOOST_CHECK(static_cast<int>(CONFIGURATION_ERROR) == 2);
	BOOST_CHECK(static_cast<int>(EVOLUTION_ERROR) == 3);
	BOOST_CHECK(static_cast<int>(NUMBER_ITERATIONS_ERROR) == 4);
	BOOST_CHECK(static_cast<int>(REPORT_ERROR) == 5);

}

void test_IsConfigured_ToggleConfigured(){
	NetworkState s = NetworkState(2.0);
	BOOST_CHECK(s._isConfigured==false);
	BOOST_CHECK(s.isConfigured()==false);
	s.toggleConfigured();
	BOOST_CHECK(s.isConfigured()==true);
	s.toggleConfigured();
	BOOST_CHECK(s.isConfigured()==false);
}

void test_GetSetResult(){
	NetworkState s = NetworkState(2.0);
	BOOST_CHECK(s._result==NOT_RUN);
	BOOST_CHECK(s.getResult()==NOT_RUN);
	s.setResult(REPORT_ERROR);
	BOOST_CHECK(s.getResult()==REPORT_ERROR);
}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	test_Constructor();
	test_enum();
	test_IsConfigured_ToggleConfigured();
	test_GetSetResult();

	return 0;
}
