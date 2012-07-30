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
#include <MPILib/include/report/handler/RootHighThroughputHandler.hpp>
#undef protected
#undef private
#include <MPILib/include/utilities/Exception.hpp>
#include <boost/test/minimal.hpp>
#include <MPILib/include/report/Report.hpp>
#include <string>
#include <memory>
#include <TGraph.h>
#include <TFile.h>
#include <TVectorD.h>
#include <TArrayI.h>

using namespace boost::unit_test;
using namespace MPILib::report::handler;
using namespace MPILib::report;

void test_Constructor() {
	RootHighThroughputHandler rH("blub");
	BOOST_CHECK(rH._streamFileName=="blub");
	BOOST_CHECK(rH._isRecording==false);
	BOOST_CHECK(rH._generateNodeGraphs==false);

	RootHighThroughputHandler rH1("blab", true);
	BOOST_CHECK(rH1._streamFileName=="blab");
	BOOST_CHECK(rH1._isRecording==false);
	BOOST_CHECK(rH1._generateNodeGraphs==true);

	RootHighThroughputHandler rHC(rH1);
	BOOST_CHECK(rHC._streamFileName=="blab");
	BOOST_CHECK(rHC._isRecording==false);
	BOOST_CHECK(rHC._generateNodeGraphs==true);

}

void test_clone() {
	RootHighThroughputHandler rH("blub");

	RootHighThroughputHandler* pRH;
	pRH = rH.clone();

	BOOST_CHECK(pRH->_streamFileName=="blub");
	BOOST_CHECK(pRH->_generateNodeGraphs==false);
	BOOST_CHECK(pRH->_isRecording==false);

	delete pRH;
}

void test_writeReport() {
	RootHighThroughputHandler rH("RootHighTestWrite", true);
	rH.initializeHandler(1);
	BOOST_CHECK(rH._pFile!=nullptr);
	ReportValue tempV { "blab", 1.0, 2.0 };

	std::vector<ReportValue> rv;
	rv.push_back(tempV);
	Report r1(1.0, 1.0, 1, MPILib::algorithm::AlgorithmGrid(2),RATE,
			rv, 3);

	BOOST_CHECK(rH._pTree == nullptr);
	BOOST_CHECK(rH._pArray == nullptr);
	BOOST_CHECK(rH._isRecording==false);
	BOOST_CHECK(rH._mData.size()==0);
	rH.writeReport(r1);
	BOOST_CHECK(rH._isRecording==true);
	BOOST_CHECK(rH._pTree != nullptr);
	BOOST_CHECK(rH._pArray != nullptr);
	BOOST_CHECK(rH._mData.size()==1);
	BOOST_CHECK(rH._mData[1]==1.0);
	auto nodeIds = std::unique_ptr<TArrayI>(
			(TArrayI*) rH._pFile->Get("GlobalNodeIds"));
	BOOST_CHECK(nodeIds->GetSize()==3);

	MPILib::utilities::MPIProxy mpiProxy;
	if (mpiProxy.getSize() == 2) {
		if (mpiProxy.getRank() == 0) {
			BOOST_CHECK((*nodeIds)[2]==4);
		} else {
			BOOST_CHECK((*nodeIds)[2]==5);
		}
	} else {
		BOOST_CHECK((*nodeIds)[2]==2);
	}

	Report r2(1.0, 2.0, 2, MPILib::algorithm::AlgorithmGrid(2), RATE,
			rv, 3);
	rH.writeReport(r2);
	BOOST_CHECK(rH._mData.size()==2);
	BOOST_CHECK(rH._mData[2]==2.0);

	Report r3(1.0, 0.0, 0, MPILib::algorithm::AlgorithmGrid(2),  RATE,
			rv, 3);
	rH.writeReport(r3);
	BOOST_CHECK(rH._mData.size()==0);
	BOOST_CHECK(rH._mData[0]==0.0);
	BOOST_CHECK((*rH._pArray)[0]==1.0);
	BOOST_CHECK((*rH._pArray)[1]==0.0);
	BOOST_CHECK((*rH._pArray)[2]==1.0);
	BOOST_CHECK((*rH._pArray)[3]==2.0);

	bool thrown = false;
	rH.writeReport(r1);
	try {
		rH.writeReport(r1);
	} catch (MPILib::utilities::Exception& e) {
		thrown = true;
	}
	BOOST_CHECK(thrown==true);

}

void test_initializeHandler() {
//it is not possible anymore to have a empty file they contain at least to extension

	RootHighThroughputHandler rH("RootTestFile", true);
	bool thrown = false;

	try {
		rH.initializeHandler(3);
	} catch (MPILib::utilities::Exception& e) {
		thrown = true;
	} catch (...) {
		BOOST_ERROR("shouble be catched already");
	}
	BOOST_CHECK(thrown==false);
	BOOST_CHECK(rH._pFile!=nullptr);
}

void test_detachHandler() {
	RootHighThroughputHandler rH("RootHighTestWrite", true);

	rH.detachHandler(1);

	BOOST_CHECK(rH._pTree == nullptr);
	BOOST_CHECK(rH._pArray == nullptr);
	BOOST_CHECK(rH._pFile == nullptr);
	BOOST_CHECK(rH._isRecording==false);
	BOOST_CHECK(rH._mData.size()==0);
}

void test_generateNodeGraphs() {
///*todo DS implement this test
}
void test_collectGraphInformation() {
///*todo DS implement this test
}
void test_storeRateGraphs() {
///*todo DS implement this test
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
	test_clone();
	test_writeReport();
	test_initializeHandler();
	test_detachHandler();
	test_generateNodeGraphs();
	test_collectGraphInformation();
	test_storeRateGraphs();

	return 0;

}
