/*
 * MPINetwork_test.cpp
 *
 *  Created on: 31.05.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <MPILib/include/BasicTypes.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/algorithm/WilsonCowanAlgorithm.hpp>
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

	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;



	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);

	BOOST_CHECK(algorithm_exc._parameter._f_noise == par_sigmoid._f_noise);
	BOOST_CHECK(
			algorithm_exc._integrator.Parameter()._f_noise == par_sigmoid._f_noise);

}

void test_clone() {
	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;


	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);

	WilsonCowanAlgorithm* alg = algorithm_exc.clone();
	BOOST_CHECK(alg->_parameter._f_noise == 1.0);
	delete alg;

	AlgorithmInterface<double>* algI;
	algI = algorithm_exc.clone();

	if (dynamic_cast<WilsonCowanAlgorithm *>(algI)) {
	} else {
		BOOST_ERROR("should be of dynamic type WilsonCowanAlgorithm");
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

	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;


	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);

	algorithm_exc.configure(PAR_WILSONCOWAN);
	BOOST_CHECK(algorithm_exc._integrator._time_begin==0);
	BOOST_CHECK(algorithm_exc._integrator._step==1e-5);

}

void test_evolveNodeState() {
	/// @todo DS implement this test
}

void test_getCurrentTime() {
	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;

	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);
	BOOST_CHECK(algorithm_exc.getCurrentTime() == 0.0);
}

void test_getCurrentRate() {
	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;


	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);
	BOOST_CHECK(algorithm_exc.getCurrentTime() == 0.0);
}

void test_innerProduct() {
	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;


	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);

	std::vector<double> weightVector = { 4.2, 5.2, 7.2 };

	std::vector<double> nodeVector = { 2.3, 5.3, 9.3 };

	//need to calculate this
	double res = algorithm_exc.innerProduct(weightVector, nodeVector);
	BOOST_CHECK(res == 104.18);

}

void test_getInitialState() {
	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;


	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm alg(par_sigmoid);

	std::vector<double> v =alg.getInitialState();
	BOOST_CHECK(v[0]==0);
}

void test_getGrid() {
	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;

	// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm alg(par_sigmoid);
	AlgorithmGrid grid = alg.getGrid();
	BOOST_CHECK(grid._arrayState[0]==0.0);
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
	test_innerProduct();
	test_getInitialState();
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
