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
#include <MPILib/include/report/handler/ValueHandlerHandler.hpp>
#undef protected
#undef private
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::report::handler;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {
	ValueHandlerHandler vh;
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==true);
	BOOST_CHECK(vh._vec_quantity.empty()==true);
	BOOST_CHECK(vh._vec_time.empty()==true);

}

void test_addReport() {
	ValueHandlerHandler vh;

	MPILib::report::Report r(1.0, 2.0, 1, "blub");
	r.addValue(MPILib::report::ReportValue{"blab", 3.0, 4.0});
	vh.addReport(r);

	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==false);
	BOOST_CHECK(vh._vec_quantity.empty()==false);
	BOOST_CHECK(vh._vec_time.empty()==false);
	BOOST_CHECK(vh._vec_names[0]=="blab-1");
	BOOST_CHECK(vh._vec_time[0][0]==4.0);
	BOOST_CHECK(vh._vec_quantity[0][0]==3.0);
}

void test_write() {
	ValueHandlerHandler vh;

	MPILib::report::Report r(1.0, 2.0, 1, "blub");
	r.addValue(MPILib::report::ReportValue{"blab", 3.0, 4.0});
	vh.addReport(r);
	vh.write();
	BOOST_CHECK(vh._is_written==true);
	//TODO improve this test

}

void test_Event() {
	ValueHandlerHandler::Event e { "blub", 1.0, 2.0 };
	BOOST_CHECK(e._str=="blub");
	BOOST_CHECK(e._time==1.0);
	BOOST_CHECK(e._value==2.0);

}

void test_isWritten() {
	ValueHandlerHandler vh;
	BOOST_CHECK(vh.isWritten()==false);
	vh._is_written = true;
	BOOST_CHECK(vh.isWritten()==true);
}

void test_reset() {
	ValueHandlerHandler vh;
	vh.distributeEvent(ValueHandlerHandler::Event { "blub", 1.0, 2.0 });
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==false);
	BOOST_CHECK(vh._vec_quantity.empty()==false);
	BOOST_CHECK(vh._vec_time.empty()==false);
	vh.reset();
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==true);
	BOOST_CHECK(vh._vec_quantity.empty()==true);
	BOOST_CHECK(vh._vec_time.empty()==true);
}

void test_distributeEvent() {
	ValueHandlerHandler vh;
	vh.distributeEvent(ValueHandlerHandler::Event { "blub", 1.0, 2.0 });
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names[0]=="blub");
	BOOST_CHECK(vh._vec_time[0][0]==1.0);
	BOOST_CHECK(vh._vec_quantity[0][0]==2.0);
	vh.distributeEvent(ValueHandlerHandler::Event { "blub", 3.0, 4.0 });
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names[0]=="blub");
	BOOST_CHECK(vh._vec_time[0][0]==1.0);
	BOOST_CHECK(vh._vec_quantity[0][0]==2.0);
	BOOST_CHECK(vh._vec_time[0][1]==3.0);
	BOOST_CHECK(vh._vec_quantity[0][1]==4.0);

}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
	// run only one one process as otherwise race conditions occure

	test_Constructor();
	test_addReport();
	test_write();
	test_Event();
	test_isWritten();
	test_reset();
	test_distributeEvent();

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
