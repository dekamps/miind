/*
 * MPINetwork_test.cpp
 *
 *  Created on: 31.05.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/algorithm/RateAlgorithm.hpp>
#undef protected
#undef private

#include <MPILib/include/report/handler/RootReportHandler.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::algorithm;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	double rate = 2.1;

	RateAlgorithm rAlg(rate);
	BOOST_CHECK(rAlg._time_current == std::numeric_limits<double>::max());
	BOOST_CHECK(rAlg._rate == 2.1);

}

void test_clone() {

	double rate = 2.1;

	RateAlgorithm rAlg(rate);

	RateAlgorithm* alg = rAlg.clone();
	BOOST_CHECK(alg->_rate == 2.1);
	delete alg;

	AlgorithmInterface<double>* algI;
	algI = rAlg.clone();

	if (dynamic_cast<RateAlgorithm *>(algI)) {
	} else {
		BOOST_ERROR("should be of dynamic type RateAlgorithm");
	}
	delete algI;
}

void test_configure() {
	const MPILib::report::handler::RootReportHandler WILSONCOWAN_HANDLER(
			"test/wilsonresponse.root", // file where the simulation results are written
			false // only rate diagrams
			);

	const MPILib::SimulationRunParameter PAR_WILSONCOWAN(WILSONCOWAN_HANDLER, // the handler object
			1000000, // maximum number of iterations
			0, // start time of simulation
			0.5, // end time of simulation
			1e-4, // report time
			1e-5, // network step time
			"test/wilsonresponse.log" // log file name
			);

	double rate = 2.1;

	RateAlgorithm rAlg(rate);
	rAlg.configure(PAR_WILSONCOWAN);
	BOOST_CHECK(rAlg._time_current==0);
}

void test_evolveNodeState() {
	double rate = 2.1;
	std::vector<double> tempVec = { 1.0 };

	RateAlgorithm rAlg(rate);

	rAlg.evolveNodeState(tempVec, tempVec, 2.1);

	BOOST_CHECK(rAlg._time_current==2.1);
}

void test_getCurrentTime() {

	double rate = 2.1;
	std::vector<double> tempVec;
	tempVec.push_back(1.0);

	RateAlgorithm rAlg(rate);
	BOOST_CHECK(rAlg.getCurrentTime()==std::numeric_limits<double>::max());

	rAlg.evolveNodeState(tempVec, tempVec, 2.1);

	BOOST_CHECK(rAlg.getCurrentTime()==2.1);
}

void test_getCurrentRate() {

	double rate = 2.1;
	RateAlgorithm rAlg(rate);
	BOOST_CHECK(rAlg.getCurrentRate() == 2.1);
}

void test_getGrid() {
	double rate = 2.1;
	RateAlgorithm rAlg(rate);
	AlgorithmGrid grid = rAlg.getGrid();

	BOOST_CHECK(grid._arrayState[0]==2.1);
	BOOST_CHECK(grid._arrayInterpretation[0]==0.0);

}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	test_Constructor();
	test_clone();
	test_configure();
	test_evolveNodeState();
	test_getCurrentTime();
	test_getCurrentRate();
	test_getGrid();

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
