/*
 * CircularDistribution_test.cpp
 *
 *  Created on: 01.06.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/utilities/CircularDistribution.hpp>
#undef protected
#undef private

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	CircularDistribution circularD;

	if (world.rank() == 0) {
		BOOST_REQUIRE(circularD._processorId==0);
		BOOST_REQUIRE(circularD._totalProcessors==world.size());
	} else if (world.rank() == 1) {
		BOOST_REQUIRE(circularD._processorId==1);
		BOOST_REQUIRE(circularD._totalProcessors==world.size());
	}

}

void test_isLocalNode() {
	CircularDistribution circularD;
	if (world.rank() == 0) {
		BOOST_REQUIRE(circularD.isLocalNode(0)==true);
		BOOST_REQUIRE(circularD.isLocalNode(1)==false);

	} else if (world.rank() == 1) {
		BOOST_REQUIRE(circularD.isLocalNode(0)==false);
		BOOST_REQUIRE(circularD.isLocalNode(1)==true);
	}
}

void test_getResponsibleProcessor() {
	CircularDistribution circularD;
	BOOST_REQUIRE(circularD.getResponsibleProcessor(1)==1);
	BOOST_REQUIRE(circularD.getResponsibleProcessor(0)==0);
}

void test_isMaster() {
	CircularDistribution circularD;
	if (world.rank() == 0) {
		BOOST_REQUIRE(circularD.isMaster()==true);
	} else {
		BOOST_REQUIRE(circularD.isMaster()==false);
	}
}

void test_getRank() {
	CircularDistribution circularD;
	BOOST_REQUIRE(circularD.getRank() == world.rank());
}

void test_getSize() {
	CircularDistribution circularD;
	BOOST_REQUIRE(circularD.getSize() == world.size());
}
int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing
	test_Constructor();
	test_isLocalNode();
	test_getResponsibleProcessor();
	test_isMaster();
	test_getRank();
	test_getSize();

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
