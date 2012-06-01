/*
 * Exception_test.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/Exception.hpp>
#include <cstring>
#include <string>
#include <iostream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

namespace mpi = boost::mpi;

mpi::communicator world;

/** Make sure we can derive from it.
 */
class TestException: public Exception {
public:
	TestException() :
			Exception("qwerty") {
	}
};

class TestExceptionWithOverload: public Exception {
public:
	TestExceptionWithOverload() :
			Exception("qwerty") {
	}
	virtual const char * what(void) {
		return Exception::what();
	}
};

void test_Constructor() {
	Exception e("message");
	BOOST_REQUIRE(strncmp("message", e.what(), 7)== 0);
	Exception e2(std::string("message"));
	BOOST_REQUIRE(strncmp("message", e2.what(), 7)== 0);
}

/** Try to instantiate a derived class.
 */
void test_DerivedClass() {
	TestException t;
	BOOST_REQUIRE(strncmp(t.what(), "qwerty", 6)== 0);
	TestExceptionWithOverload to;
	BOOST_REQUIRE(strncmp(to.what(), "qwerty", 6)== 0);
}

/** Test the macros that come with the class.
 */
void test_Macros() {
	// also test the macros
	bool thrown = false;
	try {
		miind_fail("abc");
	} catch (Exception& e) {
		BOOST_REQUIRE(strncmp("abc", e.what(), 3)== 0);
		thrown = true;
	}
	BOOST_REQUIRE(thrown== true);
	thrown = false;
	std::string abc("abc");
	try {
		miind_fail(abc);
	} catch (Exception& e) {
		BOOST_REQUIRE(strncmp("abc", e.what(), 3)== 0);
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
	test_DerivedClass();
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

