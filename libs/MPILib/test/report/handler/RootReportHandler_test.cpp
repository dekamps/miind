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
#include <MPILib/include/report/handler/RootReportHandler.hpp>
#undef protected
#undef private
#include <MPILib/include/utilities/Exception.hpp>
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::report::handler;

#include <string>
#include <TGraph.h>

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {
	RootReportHandler rH("blub");
	BOOST_CHECK(rH._streamFileName=="blub");
	BOOST_CHECK(rH._isStateWriteMandatory==false);
	BOOST_CHECK(rH._nrReports==0);

	RootReportHandler rH1("blab", true);
	BOOST_CHECK(rH1._streamFileName=="blab");
	BOOST_CHECK(rH1._isStateWriteMandatory==true);
	BOOST_CHECK(rH1._nrReports==0);

	RootReportHandler rHC(rH1);
	BOOST_CHECK(rHC._streamFileName=="blab");
	BOOST_CHECK(rHC._isStateWriteMandatory==true);
	BOOST_CHECK(rHC._nrReports==0);

}

void test_writeReport() {
	RootReportHandler rH1("RootTestWrite", true);
	BOOST_CHECK(rH1._nrReports==0);
	BOOST_CHECK(rH1._spCurrentRateGraph==nullptr);
	rH1.initializeHandler(1);
	rH1.writeReport(MPILib::report::Report (1.0, 2.0, 1, "blub" ));
	BOOST_CHECK(rH1._spCurrentRateGraph!=nullptr);
	BOOST_CHECK(std::string(rH1._spCurrentRateGraph->GetName())=="rate_1");
	double x, y;
	rH1._spCurrentRateGraph->GetPoint(0, x, y);
	BOOST_CHECK(x==1.0);
	BOOST_CHECK(y==2.0);
	BOOST_CHECK(rH1._nrReports==1);

	/// @todo DS test for state write

	rH1.finalize();

}

void test_clone() {
	RootReportHandler rH("blub");

	RootReportHandler* pRH;
	pRH = rH.clone();

	BOOST_CHECK(pRH->_streamFileName=="blub");
	BOOST_CHECK(pRH->_isStateWriteMandatory==false);
	BOOST_CHECK(pRH->_nrReports==0);

	delete pRH;
}

void test_initializeHandler() {
	//it is not possible anymore to have a empty file

	RootReportHandler rH1("RootTestFile", true);
	bool thrown = false;

	try {
		rH1.initializeHandler(3);
	} catch (MPILib::utilities::Exception& e) {
		thrown = true;
	} catch (...) {
		BOOST_ERROR("shouble be catched already");
	}
	BOOST_CHECK(thrown==false);
	BOOST_CHECK(rH1._nodes.size()==1);
}

void test_detachHandler() {
	RootReportHandler rH1("RootTestFile", true);
	rH1.initializeHandler(3);
	rH1.writeReport(MPILib::report::Report ( 1.0, 2.0, 1, "blub" ));
	BOOST_CHECK(rH1._spCurrentRateGraph!=nullptr);
	rH1.detachHandler(3);
	BOOST_CHECK(rH1._spCurrentRateGraph==nullptr);
}

void test_removeFromNodeList() {

	RootReportHandler rH1("RootTestFile", true);
	bool thrown = false;
	try {
		rH1.removeFromNodeList(2);
	} catch (MPILib::utilities::Exception& e) {
		thrown = true;
	} catch (...) {
		BOOST_ERROR("shouble be catched already");
	}
	BOOST_CHECK(thrown==true);

	thrown = false;
	try {
		BOOST_CHECK(rH1._nodes.size()==1);
		rH1.removeFromNodeList(3);
	} catch (MPILib::utilities::Exception& e) {
		thrown = true;
	} catch (...) {
		BOOST_ERROR("shouble be catched already");
	}
	BOOST_CHECK(thrown==false);
	BOOST_CHECK(rH1._nodes.size()==0);

}

void test_finalize() {
	RootReportHandler rH1("RootTestFile", true);
	rH1.initializeHandler(3);
	rH1.writeReport(MPILib::report::Report ( 1.0, 2.0, 1, "blub" ));
	BOOST_CHECK(rH1._pFile!=nullptr);
	rH1.finalize();
	BOOST_CHECK(rH1._nodes.empty()==true);
	BOOST_CHECK(rH1._pFile==nullptr);

}

void test_convertAlgorithmGridToGraph() {
	/// *todo DS implement this test
}

void test_isConnectedToAlgorithm() {
	RootReportHandler rH1("blab", true);
	BOOST_CHECK(rH1.isConnectedToAlgorithm()==false);
	rH1.writeReport(MPILib::report::Report ( 1.0, 2.0, 1, "blub" ));
	BOOST_CHECK(rH1.isConnectedToAlgorithm()==true);

}

void test_isStateWriteMandatory() {
	RootReportHandler rH1("blab", true);
	BOOST_CHECK(rH1.isStateWriteMandatory()==true);
	rH1._isStateWriteMandatory = false;
	BOOST_CHECK(rH1.isStateWriteMandatory()==false);
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
	test_writeReport();
	test_clone();
	test_initializeHandler();
	test_detachHandler();
	test_removeFromNodeList();
	test_finalize();
	test_convertAlgorithmGridToGraph();
	test_isConnectedToAlgorithm();
	test_isStateWriteMandatory();

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
