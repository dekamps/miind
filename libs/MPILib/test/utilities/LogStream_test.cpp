/*
 * TimeException_test.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#define private public
#define protected public
#include <MPILib/include/utilities/LogStream.hpp>
#undef protected
#undef private
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

	LogStream ls;

	BOOST_CHECK(ls._isTimeAvailable == true);

	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls2(os);
	BOOST_CHECK(ls2._isTimeAvailable == true);
	BOOST_CHECK(ls2._pStreamLog == os);

}

void test_Destructor() {
	std::shared_ptr<std::ostream> os1(new std::ostringstream);
	LogStream* ls3 = new LogStream(os1);
	delete ls3;

	std::stringstream temp;
	temp << "Total time: ";
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os1)->str().compare(0, temp.str().length(), temp.str())==0);
}

void test_Record() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	ls.record("blub");
	std::string s = "blub";
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s)!= std::string::npos);
}

void test_GetStream() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(ls.getStream()==os);
}

void test_OpenStream() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(ls.openStream(os)==false);

	LogStream ls1;
	BOOST_CHECK(ls1.openStream(os)==true);
}

void test_Operators() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	std::stringstream s;
	s << std::string("blub");
	ls << std::string("blub");
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);
	s.clear();
	s << "blub2";
	ls << "blub2";
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);
	s.clear();
	s << int(1);
	ls << int(1);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);
	s.clear();
	s << double(1.4);
	ls << double(1.4);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);

}

int test_main(int argc, char* argv[]) // note the name!
		{

	mpi::environment env(argc, argv);

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	// we use only two processors for this testing
	test_Constructor();
	test_Destructor();
	test_Record();
	test_GetStream();
	test_OpenStream();
	test_Operators();

	return 0;

}

