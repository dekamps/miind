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
	BOOST_CHECK(strncmp("message", e.what(), 7)== 0);
	Exception e2(std::string("message"));
	BOOST_CHECK(strncmp("message", e2.what(), 7)== 0);
}

/** Try to instantiate a derived class.
 */
void test_DerivedClass() {
	TestException t;
	BOOST_CHECK(strncmp(t.what(), "qwerty", 6)== 0);
	TestExceptionWithOverload to;
	BOOST_CHECK(strncmp(to.what(), "qwerty", 6)== 0);
}

/** Test the macros that come with the class.
 */
void test_Macros() {
	// also test the macros
	bool thrown = false;
	try {
		miind_fail("abc");
	} catch (Exception& e) {
		BOOST_CHECK(strncmp("abc", e.what(), 3)== 0);
		thrown = true;
	}
	BOOST_CHECK(thrown== true);
	thrown = false;
	std::string abc("abc");
	try {
		miind_fail(abc);
	} catch (Exception& e) {
		BOOST_CHECK(strncmp("abc", e.what(), 3)== 0);
		thrown = true;
	}
	BOOST_CHECK(thrown== true);
}

void test_catch() {

	try {
		throw Exception("message");
	} catch (Exception& e) {

	} catch (...) {
		BOOST_FAIL("should be catched already");

	}

	try {
		throw Exception("message");
	} catch (std::exception& e) {

	} catch (...) {
		BOOST_FAIL("should be catched already");

	}



	try {
		throw Exception("message");
	} catch (Exception& e) {

	} catch (std::exception& e) {
		BOOST_FAIL("should be catched already");
	}

	try {
		throw std::exception();
	} catch (Exception& e) {
		BOOST_FAIL("should not be catched as it is a Exception");

	} catch (...) {
	}

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
	test_catch();

	return 0;
//    // six ways to detect and report the same error:
//    BOOST_CHECK( add( 2,2 ) == 4 );        // #1 continues on error
//    BOOST_CHECK( add( 2,2 ) == 4 );      // #2 throws on error
//    if( add( 2,2 ) != 4 )
//        BOOST_ERROR( "Ouch..." );          // #3 continues on error
//    if( add( 2,2 ) != 4 )
//        BOOST_FAIL( "Ouch..." );           // #4 throws on error
//    if( add( 2,2 ) != 4 ) throw "Oops..."; // #5 throws on error
//
//    return add( 2, 2 ) == 4 ? 0 : 1;       // #6 returns error code
}

