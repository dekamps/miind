/*
 * MPINetwork_test.cpp
 *
 *  Created on: 31.05.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/algorithm/RateAlgorithm.hpp>
#undef protected
#undef private

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::algorithm;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	double rate = 2.1;

	RateAlgorithm rAlg(rate);
	BOOST_REQUIRE(rAlg._time_current == numeric_limits<double>::max());
	BOOST_REQUIRE(rAlg._rate == 2.1);

}

void test_clone() {
	//TODO
}

void test_configure() {
	//TODO
}

void test_evolveNodeState() {
	double rate = 2.1;
	std::vector<double> tempVec = {1.0};

	RateAlgorithm rAlg(rate);

	rAlg.evolveNodeState(tempVec, tempVec, 2.1);

	BOOST_REQUIRE(rAlg._time_current==2.1);
}

void test_getCurrentTime() {

	double rate = 2.1;
	std::vector<double> tempVec;
	tempVec.push_back(1.0);

	RateAlgorithm rAlg(rate);
	BOOST_REQUIRE(rAlg.getCurrentTime()==numeric_limits<double>::max());

	rAlg.evolveNodeState(tempVec, tempVec, 2.1);

	BOOST_REQUIRE(rAlg.getCurrentTime()==2.1);
}

void test_getCurrentRate() {

	double rate = 2.1;
	RateAlgorithm rAlg(rate);
	BOOST_REQUIRE(rAlg.getCurrentRate() == 2.1);
}

void test_getGrid() {
	//TODO
}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	test_Constructor();
	test_clone();
	test_configure();
	test_evolveNodeState();
	test_getCurrentTime();
	test_getCurrentRate();
	test_getGrid();

	return 0;
//    // six ways to detect and report the same error:
//    BOOST_CHECK( add( 2,2 ) == 4 );        // #1 continues on error
//    BOOST_REQUIRE( add( 2,2 ) == 4 );      // #2 throws on error
//    if( add( 2,2 ) != 4 )
//        BOOST_ERROR( "Ouch..." );          // #3 continues on error
//    if( add( 2,2 ) != 4 )
//        BOOST_FAIL( "Ouch..." );           // #4 throws on error
//    if( add( 2,2 ) != 4 ) throw "Oops..."; // #5 throws on error
//
//    return add( 2, 2 ) == 4 ? 0 : 1;       // #6 returns error code
}
