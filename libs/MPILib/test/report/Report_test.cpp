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
#include <MPILib/include/report/Report.hpp>
#undef protected
#undef private
#include <MPILib/include/report/ReportType.hpp>

#include <MPILib/include/algorithm/AlgorithmGrid.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::report;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {
	Report r(1.0, 2.0, 1, "blub");

	BOOST_CHECK(r._id ==1);
	BOOST_CHECK(r._log_message=="blub");
	BOOST_CHECK(r._rate==2.0);
	BOOST_CHECK(r._time==1.0);

	std::vector<ReportValue> rv {ReportValue{"blab", 1.0, 2.0}};
	Report r1(1.0, 2.0, 1,  MPILib::algorithm::AlgorithmGrid(2), "blub", RATE, rv);

	BOOST_CHECK(r1._id==1);
	BOOST_CHECK(r1._log_message=="blub");
	BOOST_CHECK(r1._grid.getStateSize()==2);
	BOOST_CHECK(r1._rate==2.0);
	BOOST_CHECK(r1._time==1.0);
	BOOST_CHECK(r1._type==RATE);
	BOOST_CHECK(r1._values[0]._name_quantity=="blab");
	BOOST_CHECK(r1._nrNodes == 0);

	Report r2(1.0, 2.0, 1,  MPILib::algorithm::AlgorithmGrid(2), "blub", RATE, rv, 10);

	BOOST_CHECK(r2._id==1);
	BOOST_CHECK(r2._log_message=="blub");
	BOOST_CHECK(r2._grid.getStateSize()==2);
	BOOST_CHECK(r2._rate==2.0);
	BOOST_CHECK(r2._time==1.0);
	BOOST_CHECK(r2._type==RATE);
	BOOST_CHECK(r2._values[0]._name_quantity=="blab");
	BOOST_CHECK(r2._nrNodes == 10);
}

void test_addValue(){
	Report r(1.0, 2.0, 1, "blub");

	r.addValue(ReportValue{"blab", 1.0, 2.0});
	BOOST_CHECK(r._values[0]._name_quantity=="blab");

}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	test_Constructor();
	test_addValue();
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
