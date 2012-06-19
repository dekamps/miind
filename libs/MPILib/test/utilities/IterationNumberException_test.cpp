/*
 * IterationNumberException_test.cpp
 *
 *  Created on: 19.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/IterationNumberException.hpp>
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
	IterationNumberException e("message");
	std::stringstream sstream;
	sstream << "message";

	BOOST_REQUIRE(
			strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);
	IterationNumberException e2(std::string("message"));
	BOOST_REQUIRE(
			strncmp(sstream.str().c_str(), e2.what(), sstream.str().size())== 0);
}

void test_catch() {

	try {
		throw IterationNumberException("message");
	} catch (IterationNumberException& e) {
		std::stringstream sstream;
		sstream << "message";
		BOOST_REQUIRE(
				strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);

	}

	try {
		throw IterationNumberException("message");
	} catch (Exception& e) {
		std::stringstream sstream;
		sstream << "message";
		BOOST_REQUIRE(
				strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);

	}

	try {
		throw IterationNumberException("message");
	} catch (std::exception& e) {
		std::stringstream sstream;
		sstream << "message";
		BOOST_REQUIRE(
				strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);

	}

	try {
		throw IterationNumberException("message");
	} catch (IterationNumberException& e) {

	} catch (std::exception& e) {
		BOOST_FAIL("should be catched already");
	}

	try {
		throw Exception("message");
	} catch (IterationNumberException& e) {
		BOOST_FAIL("should not be catched as it is a Exception");

	} catch (Exception& e) {
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
	test_catch();

	return 0;

}

