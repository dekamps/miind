/*
 * TimeException_test.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#define private public
#define protected public
#include <MPILib/include/utilities/ProgressBar.hpp>
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

	std::stringstream os;

	ProgressBar pb(100, "blub", os);

	if (world.rank() == 0) {
		BOOST_CHECK(pb._expectedCount==100);
	}
	BOOST_CHECK(pb._description=="blub");

	if (world.rank() == 0) {

		std::stringstream tempStream;
		tempStream << "blub" << "\n"
				<< "0%   10   20   30   40   50   60   70   80   90   100%\n"
				<< "|----|----|----|----|----|----|----|----|----|----|"
				<< std::endl;
		BOOST_CHECK(os.str()==tempStream.str());
		pb++;
		pb++;
		pb++;
		pb++;
		tempStream << "***";
		BOOST_CHECK(os.str()==tempStream.str());
		++pb;
		++pb;
		++pb;
		tempStream << "*";
		BOOST_CHECK(os.str()==tempStream.str());
		pb+=50;
		tempStream <<"************************";
		BOOST_CHECK(os.str()==tempStream.str());

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

	return 0;

}

