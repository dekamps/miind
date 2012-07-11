/*
 * Report_test.cpp
 *
 *  Created on: 21.06.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/report/handler/GraphKey.hpp>
#undef protected
#undef private
#include <string>
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::report::handler;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_enum(){
	BOOST_CHECK(RATEGRAPH==1);
	BOOST_CHECK(STATEGRAPH==0);
}

void test_Constructor() {
	GraphKey gk;

	BOOST_CHECK(gk._id==0);
	BOOST_CHECK(gk._time==0.0);
	BOOST_CHECK(gk._type==STATEGRAPH);

	GraphKey gk2(1, 2.0);
	BOOST_CHECK(gk2._id==1);
	BOOST_CHECK(gk2._time==2.0);
	BOOST_CHECK(gk2._type==STATEGRAPH);

	std::string key = "grid_3_3.0";
	GraphKey gk3(key);
	BOOST_CHECK(gk3._id==3);
	BOOST_CHECK(gk3._time==3.0);
	BOOST_CHECK(gk3._type==STATEGRAPH);

	std::string key1 = "rate_4";
	GraphKey gk4(key1);
	BOOST_CHECK(gk4._id==4);
	BOOST_CHECK(gk4._time==0.0);
	BOOST_CHECK(gk4._type==RATEGRAPH);

}

void test_generateName() {
	std::string key = "rate_4";
	GraphKey gk(key);

	BOOST_CHECK(gk.generateName()=="rate_4");

	std::string key1 = "grid_3_3.0";
	GraphKey gk1(key1);
	BOOST_CHECK(gk1.generateName()=="grid_3_3");
}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
	test_enum();
	test_Constructor();
	test_generateName();
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
