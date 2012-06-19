/*
 * TimeException_test.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/Timer.hpp>
#include <ctime>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Timer() {

	Timer t;

	//wait for 0.1 sec
	time_t start_time, cur_time;

	time(&start_time);
	do {
		time(&cur_time);
	} while ((cur_time - start_time) < 0.1);

	BOOST_REQUIRE(
			t.secondsSinceLastCall()>0.09 && t.secondsSinceLastCall()<0.11);

	time(&start_time);
	do {
		time(&cur_time);
	} while ((cur_time - start_time) < 0.1);

	BOOST_REQUIRE(
			t.secondsSinceLastCall()>0.09 && t.secondsSinceLastCall()<0.11);
	BOOST_REQUIRE( t.secondsSinceFirstCall()>0.19);

}

int test_main(int argc, char* argv[]) // note the name!
		{

	mpi::environment env(argc, argv);

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	// we use only two processors for this testing
	test_Timer();

	return 0;

}

