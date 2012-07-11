/*
 * MPIProxy_test.cpp
 *
 *  Created on: 11.07.2012
 *      Author: david
 */

#define private public
#define protected public
#include <MPILib/include/utilities/MPIProxy.hpp>
#undef protected
#undef private

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	MPIProxy mpiProxy;

	BOOST_CHECK(mpiProxy._rank == world.rank());
	BOOST_CHECK(mpiProxy._size == world.size());
}

void test_Getter() {
	MPIProxy mpiProxy;
	BOOST_CHECK(mpiProxy.getRank() == world.rank());
	BOOST_CHECK(mpiProxy.getSize() == world.size());

}

void test_Broadcast(){

	int blub = 0;
	BOOST_CHECK(blub == 0);
	if(world.rank()==0){
		blub = 9;
		BOOST_CHECK(blub == 9);
	}
	MPIProxy mpiProxy;
	mpiProxy.broadcast(blub, 0);

	BOOST_CHECK(blub == 9);

	blub = 0;
	BOOST_CHECK(blub == 0);
	if(world.rank()==0){
		blub = 9;
		BOOST_CHECK(blub == 9);
	}
	mpiProxy.broadcast(blub, 1);

	BOOST_CHECK(blub == 0);
}

int test_main(int argc, char* argv[]) // note the name!
		{

	mpi::environment env(argc, argv);

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	// we use only two processors for this testing
	test_Constructor();
	test_Getter();
	test_Broadcast();

	return 0;

}

