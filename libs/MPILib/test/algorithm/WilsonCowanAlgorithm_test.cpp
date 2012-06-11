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

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::algorithm;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;

	// define some efficacy
	MPILib::Efficacy epsilon = 1.0;

	// define some input rate
	MPILib::Rate nu = 0;

	// Define the receiving node
	DynamicLib::WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);

	BOOST_REQUIRE(algorithm_exc._parameter._f_noise == par_sigmoid._f_noise);
	BOOST_REQUIRE(
			algorithm_exc._integrator.Parameter()._f_noise == par_sigmoid._f_noise);

}

void test_clone() {
	//TODO
}

void test_configure() {
	//TODO
}

void test_evolveNodeState() {
	//TODO
}

void test_getCurrentTime() {
//TODO
}

void test_getCurrentRate() {

	//TODO
}

void test_innerProduct() {
	MPILib::Time tau = 10e-3; //10 ms
	MPILib::Rate rate_max = 100.0;
	double noise = 1.0;

	// define some efficacy
	MPILib::Efficacy epsilon = 1.0;

	// define some input rate
	MPILib::Rate nu = 0;

	// Define the receiving node
	DynamicLib::WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);

	std::vector<double> weightVector = { 4.2, 5.2, 7.2 };

	std::vector<double> nodeVector = { 2.3, 5.3, 9.3 };

	//need to calculate this
	double res = algorithm_exc.innerProduct(weightVector, nodeVector);
	BOOST_REQUIRE(res == 104.18);

}

void test_getInitialState() {
	//TODO
}

void test_getGrid() {
	//TODO
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
//    BOOST_REQUIRE( add( 2,2 ) == 4 );      // #2 throws on error
//    if( add( 2,2 ) != 4 )
//        BOOST_ERROR( "Ouch..." );          // #3 continues on error
//    if( add( 2,2 ) != 4 )
//        BOOST_FAIL( "Ouch..." );           // #4 throws on error
//    if( add( 2,2 ) != 4 ) throw "Oops..."; // #5 throws on error
//
//    return add( 2, 2 ) == 4 ? 0 : 1;       // #6 returns error code
}
