/*
 * Log_test.cpp
 *
 *  Created on: 13.07.2012
 *      Author: david
 */

#define private public
#define protected public
#include <MPILib/include/utilities/Log.hpp>
#undef protected
#undef private
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
namespace mpi = boost::mpi;

mpi::communicator world;
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
namespace MPILib {
namespace utilities {

void test_Constructor() {
	Log lg;
}

void test_Destructor() {
	Log* lg = new Log();
	std::ostringstream* pStream = new std::ostringstream();

	Log::getStream() = pStream;

	lg->writeReport() << "blub";
	BOOST_CHECK(pStream->str().find("blub")!=26);
	delete lg;
	BOOST_CHECK(pStream->str().find("blub")==26);

}

void test_LogLevel() {
	BOOST_CHECK(logERROR<logWARNING);
	BOOST_CHECK(logWARNING<logINFO);
	BOOST_CHECK(logINFO<logDEBUG);
	BOOST_CHECK(logDEBUG<logDEBUG1);
	BOOST_CHECK(logDEBUG1<logDEBUG2);
	BOOST_CHECK(logDEBUG2<logDEBUG3);
	BOOST_CHECK(logDEBUG3<logDEBUG4);

}

void test_writeReport() {
	Log lg;

	lg.writeReport() << "blub" << 42;
	BOOST_CHECK(lg._buffer.str().find("blub")==26);
	BOOST_CHECK(lg._buffer.str().find("42")==30);

}

void test_getStream() {
	Log* lg = new Log();
	std::ostringstream* pStream = new std::ostringstream();

	Log::getStream() = pStream;

	BOOST_CHECK(pStream == Log::getStream());
	delete lg;
}

void test_writeOutput() {
	std::ostringstream* pStream = new std::ostringstream();

	Log::getStream() = pStream;
	Log::writeOutput(std::string("blub"));
	BOOST_CHECK(pStream->str().find("blub")==0);

}

void test_getReportingLevel() {
	BOOST_CHECK(Log::getReportingLevel()==logDEBUG4);
}

void test_setReportingLevel() {
	BOOST_CHECK(Log::getReportingLevel()==logDEBUG4);
	std::ostringstream* pStream = new std::ostringstream();

	Log::getStream() = pStream;
	Log::setReportingLevel(logERROR);
	BOOST_CHECK(pStream->str().find("Report")==26);
	BOOST_CHECK(Log::getReportingLevel()==logERROR);

}

void test_Macro() {
	std::ostringstream* pStream = new std::ostringstream();

	Log::getStream() = pStream;

	LOG(logERROR) << "blub" << 42;

	BOOST_CHECK(pStream->str().find("blub")==27);
	BOOST_CHECK(pStream->str().find("42")==31);
	Log::setReportingLevel(logERROR);
	pStream = new std::ostringstream();

	Log::getStream() = pStream;
	LOG(logDEBUG) << "blub" << 42;
	BOOST_CHECK(pStream->str().find("blub")!=27);
	BOOST_CHECK(pStream->str().find("42")!=31);
	pStream = new std::ostringstream();

	Log::getStream() = pStream;
	LOG(logERROR) << "blub" << 42;

	BOOST_CHECK(pStream->str().find("blub")==27);
	BOOST_CHECK(pStream->str().find("42")==31);
}

}
}

int test_main(int argc, char* argv[]) // note the name!
		{

	mpi::environment env(argc, argv);
	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

// we use only two processors for this testing
	MPILib::utilities::test_Constructor();
	MPILib::utilities::test_Destructor();
	MPILib::utilities::test_LogLevel();
	MPILib::utilities::test_writeReport();
	MPILib::utilities::test_getStream();
	MPILib::utilities::test_writeOutput();
	MPILib::utilities::test_getReportingLevel();
	MPILib::utilities::test_setReportingLevel();
	MPILib::utilities::test_Macro();

	return 0;

}

