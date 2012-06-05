
#include <iostream>
#include <MPILib/include/utilities/Time.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

namespace mpi = boost::mpi;

mpi::communicator world;

/** Test free operators
 */
void testFreeOperators() {
	timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	tv2 = tv1;
	BOOST_REQUIRE(operator==(tv1, tv2)== true);
	BOOST_REQUIRE(operator==(tv2, tv1)== true);
	BOOST_REQUIRE(operator<=(tv1, tv2)== true);
	BOOST_REQUIRE(operator<=(tv2, tv1)== true);
	tv2.tv_sec += 1;
	BOOST_REQUIRE(operator==(tv1, tv2)== false);
	BOOST_REQUIRE(operator==(tv2, tv1)== false);
	BOOST_REQUIRE(operator<=(tv1, tv2)== true);
	BOOST_REQUIRE(operator<=(tv2, tv1)== false);
	MPILib::utilities::set_max(tv2.tv_sec);
	MPILib::utilities::set_max(tv2.tv_usec);
	BOOST_REQUIRE(operator==(tv1, tv2)== false);
	BOOST_REQUIRE(operator==(tv2, tv1)== false);
	BOOST_REQUIRE(operator<=(tv1, tv2)== true);
	BOOST_REQUIRE(operator<=(tv2, tv1)== false);
	MPILib::utilities::set_max(tv1.tv_sec);
	MPILib::utilities::set_max(tv1.tv_usec);
	BOOST_REQUIRE(operator==(tv1, tv2)== true);
	BOOST_REQUIRE(operator==(tv2, tv1)== true);
	BOOST_REQUIRE(operator<=(tv1, tv2)== true);
	BOOST_REQUIRE(operator<=(tv2, tv1)== true);
	MPILib::utilities::set_min(tv2.tv_sec);
	MPILib::utilities::set_min(tv2.tv_usec);
	BOOST_REQUIRE(operator==(tv1, tv2)== false);
	BOOST_REQUIRE(operator==(tv2, tv1)== false);
	BOOST_REQUIRE(operator<=(tv1, tv2)== false);
	BOOST_REQUIRE(operator<=(tv2, tv1)== true);
}

int test_main(int argc, char* argv[]) // note the name!
		{

	mpi::environment env(argc, argv);

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	testFreeOperators();
	return 0;

}

