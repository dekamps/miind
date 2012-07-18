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

#include <MPILib/include/utilities/IterationNumberException.hpp>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;


void test_Constructor() {
	IterationNumberException e("message");
	std::stringstream sstream;
	sstream << "message";

	BOOST_CHECK(
			strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);
	IterationNumberException e2(std::string("message"));
	BOOST_CHECK(
			strncmp(sstream.str().c_str(), e2.what(), sstream.str().size())== 0);
}

void test_catch() {

	try {
		throw IterationNumberException("message");
	} catch (IterationNumberException& e) {
		std::stringstream sstream;
		sstream << "message";
		BOOST_CHECK(
				strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);

	}

	try {
		throw IterationNumberException("message");
	} catch (Exception& e) {
		std::stringstream sstream;
		sstream << "message";
		BOOST_CHECK(
				strncmp(sstream.str().c_str(), e.what(), sstream.str().size())== 0);

	}

	try {
		throw IterationNumberException("message");
	} catch (std::exception& e) {
		std::stringstream sstream;
		sstream << "message";
		BOOST_CHECK(
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

	// we use only two processors for this testing
	test_Constructor();
	test_catch();

	return 0;

}

