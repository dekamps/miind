/*
 * SimulationRunParameter_test.cpp
 *
 *  Created on: 19.06.2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

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

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor_and_Copy() {
	report::handler::InactiveReportHandler handler;

	SimulationRunParameter simParam { handler, 0, 0.0, 0.0, 0.0, 0.0, "" };

	BOOST_CHECK(simParam._p_handler == &handler);
	BOOST_CHECK(simParam._max_iter == 0);
	BOOST_CHECK(simParam._t_begin ==0.0);
	BOOST_CHECK(simParam._t_end==0.0);
	BOOST_CHECK(simParam._t_report==0.0);
	BOOST_CHECK(simParam._t_step==0.0);
	BOOST_CHECK(simParam._name_log=="");
	BOOST_CHECK(simParam._t_state_report == 0.0);

	SimulationRunParameter simParam2 { report::handler::InactiveReportHandler(),
			1, 1.0, 1.0, 1.0, 1.0, "a", 2.0 };

	BOOST_CHECK(simParam2._p_handler != &handler);
	BOOST_CHECK(simParam2._max_iter == 1);
	BOOST_CHECK(simParam2._t_begin ==1.0);
	BOOST_CHECK(simParam2._t_end==1.0);
	BOOST_CHECK(simParam2._t_report==1.0);
	BOOST_CHECK(simParam2._t_step==1.0);
	BOOST_CHECK(simParam2._name_log=="a");
	BOOST_CHECK(simParam2._t_state_report == 2.0);

	SimulationRunParameter simParam3 { simParam2 };

	BOOST_CHECK(simParam3._p_handler != &handler);
	BOOST_CHECK(simParam3._max_iter == 1);
	BOOST_CHECK(simParam3._t_begin ==1.0);
	BOOST_CHECK(simParam3._t_end==1.0);
	BOOST_CHECK(simParam3._t_report==1.0);
	BOOST_CHECK(simParam3._t_step==1.0);
	BOOST_CHECK(simParam3._name_log=="a");
	BOOST_CHECK(simParam3._t_state_report == 2.0);

	SimulationRunParameter simParam4 = simParam2;

	BOOST_CHECK(simParam4._p_handler != &handler);
	BOOST_CHECK(simParam4._max_iter == 1);
	BOOST_CHECK(simParam4._t_begin ==1.0);
	BOOST_CHECK(simParam4._t_end==1.0);
	BOOST_CHECK(simParam4._t_report==1.0);
	BOOST_CHECK(simParam4._t_step==1.0);
	BOOST_CHECK(simParam4._name_log=="a");
	BOOST_CHECK(simParam4._t_state_report == 2.0);
}

void test_Getters() {
	report::handler::InactiveReportHandler handler;

	SimulationRunParameter simParam2 { handler,
			1, 1.0, 1.0, 1.0, 1.0, "a", 2.0 };

	BOOST_CHECK(&simParam2.getHandler() == &handler);
	BOOST_CHECK(simParam2.getMaximumNumberIterations() == 1);
	BOOST_CHECK(simParam2.getTBegin() ==1.0);
	BOOST_CHECK(simParam2.getTEnd()==1.0);
	BOOST_CHECK(simParam2.getTReport()==1.0);
	BOOST_CHECK(simParam2.getTStep()==1.0);
	BOOST_CHECK(simParam2.getLogName()=="a");
	BOOST_CHECK(simParam2.getTState() == 2.0);
}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	test_Constructor_and_Copy();
	test_Getters();

	return 0;
}
