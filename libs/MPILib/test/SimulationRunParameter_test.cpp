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
#include <MPILib/include/SimulationRunParameter.hpp>
#undef protected
#undef private
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib;



void test_Constructor_and_Copy() {
	report::handler::InactiveReportHandler handler;

	SimulationRunParameter simParam ( handler, 0, 0.0, 0.0, 0.0, 0.0, "" );

	BOOST_CHECK(simParam._pHandler == &handler);
	BOOST_CHECK(simParam._maxIter == 0);
	BOOST_CHECK(simParam._tBegin ==0.0);
	BOOST_CHECK(simParam._tEnd==0.0);
	BOOST_CHECK(simParam._tReport==0.0);
	BOOST_CHECK(simParam._tStep==0.0);
	BOOST_CHECK(simParam._logFileName=="");
	BOOST_CHECK(simParam._tStateReport == 0.0);

	SimulationRunParameter simParam2 ( report::handler::InactiveReportHandler(),
			1, 1.0, 1.0, 1.0, 1.0, "a", 2.0 );

	BOOST_CHECK(simParam2._pHandler != &handler);
	BOOST_CHECK(simParam2._maxIter == 1);
	BOOST_CHECK(simParam2._tBegin ==1.0);
	BOOST_CHECK(simParam2._tEnd==1.0);
	BOOST_CHECK(simParam2._tReport==1.0);
	BOOST_CHECK(simParam2._tStep==1.0);
	BOOST_CHECK(simParam2._logFileName=="a");
	BOOST_CHECK(simParam2._tStateReport == 2.0);

	SimulationRunParameter simParam3 ( simParam2 );

	BOOST_CHECK(simParam3._pHandler != &handler);
	BOOST_CHECK(simParam3._maxIter == 1);
	BOOST_CHECK(simParam3._tBegin ==1.0);
	BOOST_CHECK(simParam3._tEnd==1.0);
	BOOST_CHECK(simParam3._tReport==1.0);
	BOOST_CHECK(simParam3._tStep==1.0);
	BOOST_CHECK(simParam3._logFileName=="a");
	BOOST_CHECK(simParam3._tStateReport == 2.0);

	SimulationRunParameter simParam4 = simParam2;

	BOOST_CHECK(simParam4._pHandler != &handler);
	BOOST_CHECK(simParam4._maxIter == 1);
	BOOST_CHECK(simParam4._tBegin ==1.0);
	BOOST_CHECK(simParam4._tEnd==1.0);
	BOOST_CHECK(simParam4._tReport==1.0);
	BOOST_CHECK(simParam4._tStep==1.0);
	BOOST_CHECK(simParam4._logFileName=="a");
	BOOST_CHECK(simParam4._tStateReport == 2.0);
}

void test_Getters() {
	report::handler::InactiveReportHandler handler;

	SimulationRunParameter simParam2 ( handler,
			1, 1.0, 1.0, 1.0, 1.0, "a", 2.0 );

	BOOST_CHECK(&simParam2.getHandler() == &handler);
	BOOST_CHECK(simParam2.getMaximumNumberIterations() == 1);
	BOOST_CHECK(simParam2.getTBegin() ==1.0);
	BOOST_CHECK(simParam2.getTEnd()==1.0);
	BOOST_CHECK(simParam2.getTReport()==1.0);
	BOOST_CHECK(simParam2.getTStep()==1.0);

	if (MPILib::utilities::MPIProxySingleton::instance().getRank()==0){
		BOOST_CHECK(simParam2.getLogName()=="a_0.log");
	}else{
		BOOST_CHECK(simParam2.getLogName()=="a_1.log");
	}
	BOOST_CHECK(simParam2.getTState() == 2.0);
}

int test_main(int argc, char* argv[]) // note the name!
		{
#ifdef ENABLE_MPI
	boost::mpi::environment env(argc, argv);

	// we use only two processors for this testing
	if (MPILib::utilities::MPIProxySingleton::instance().getSize() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
#endif
	test_Constructor_and_Copy();
	test_Getters();

	return 0;
}
