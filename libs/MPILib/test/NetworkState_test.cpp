/*
 * NetworkState.cpp
 *
 *  Created on: 19.06.2012
 *      Author: david
 */

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
