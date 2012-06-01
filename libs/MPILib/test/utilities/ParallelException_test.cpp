/*
 * Exception_test.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/ParallelException.hpp>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {
	ParallelException e("message");
	std::stringstream sstream;
	sstream << std::endl << "Parallel Exception on processor: " << world.rank()
			<< " from: " << world.size() << " with error message: message"
			<< std::endl;

	BOOST_REQUIRE(
			strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);
	ParallelException e2(std::string("message"));
	BOOST_REQUIRE(
			strncmp(sstream.str().c_str(), e2.what(), sstream.str().size())== 0);
}

/** Test the macros that come with the class.
 */
void test_Macros() {
	// also test the macros
	bool thrown = false;
	std::stringstream sstream;
	sstream << std::endl << "Parallel Exception on processor: " << world.rank()
			<< " from: " << world.size() << " with error message: abc"
			<< std::endl;

	try {
		miind_parallel_fail("abc");
	} catch (Exception& e) {
		BOOST_REQUIRE(
				strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);
		thrown = true;
	}
	BOOST_REQUIRE(thrown== true);
	thrown = false;
	std::string abc("abc");
	try {
		miind_parallel_fail(abc);
	} catch (Exception& e) {
		BOOST_REQUIRE(
				strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);
		thrown = true;
	}
	BOOST_REQUIRE(thrown== true);
}

int test_main(int argc, char* argv[]) // note the name!
		{

	mpi::environment env(argc, argv);

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	// we use only two processors for this testing
	test_Constructor();
	test_Macros();

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

