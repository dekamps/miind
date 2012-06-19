/*
 * NodeType_test.cpp
 *
 *  Created on: 19.06.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/NodeType.hpp>
#undef protected
#undef private

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_enum(){
	BOOST_REQUIRE(static_cast<int>(NodeType::NEUTRAL) == 0);
	BOOST_REQUIRE(static_cast<int>(NodeType::EXCITATORY) == 1);
	BOOST_REQUIRE(static_cast<int>(NodeType::INHIBITORY) == 2);
	BOOST_REQUIRE(static_cast<int>(NodeType::EXCITATORY_BURST) == 3);
	BOOST_REQUIRE(static_cast<int>(NodeType::INHIBITORY_BURST) == 4);

}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	test_enum();

	return 0;
}
