// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#include <MPILib/config.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
#endif
#include <MPILib/include/utilities/MPIProxy.hpp>


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
	rH1.writeReport(MPILib::report::Report (1.0, 2.0, 1));
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
	rH1.writeReport(MPILib::report::Report ( 1.0, 2.0, 1));
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
	rH1.writeReport(MPILib::report::Report ( 1.0, 2.0, 1));
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
	rH1.writeReport(MPILib::report::Report ( 1.0, 2.0, 1));
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

#ifdef ENABLE_MPI
	boost::mpi::environment env(argc, argv);
	MPILib::utilities::MPIProxy mpiProxy;

	// we use only two processors for this testing
	if (mpiProxy.getSize() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
#endif
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
