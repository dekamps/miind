#include <boost/python.hpp>
#include <vector>
#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/DelayAlgorithmCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/report/handler/AbstractReportHandler.hpp>

#ifndef MPILIB_MIINDTVBMODELABSTRACT_HPP_
#define MPILIB_MIINDTVBMODELABSTRACT_HPP_

namespace MPILib {

template<class Weight, class NodeDistribution>
class MiindTvbModelAbstract {
public:

	MiindTvbModelAbstract(int num_nodes, long simulation_length) :
		_num_nodes(num_nodes), _simulation_length(simulation_length){
	}

	~MiindTvbModelAbstract() {
		endSimulation();
	}

	virtual void init(boost::python::list) {};

	int startSimulation() {
		pb = new utilities::ProgressBar(network.startSimulation());

		// child processes just loop here - evolve isn't called here to avoid a deadlock
		// issue - would like to fix.

		if(utilities::MPIProxy().getRank() > 0)
			for(int i=0; i<int(_simulation_length/_time_step)+1; i++)
				evolveSingleStep(boost::python::list());

		// Return the MPI process rank so that the host python program can
		// end child processes after evolve (otherwise, it'll continue to try to
	  // generate a new TVB simulation which it already did in rank 0)
		return utilities::MPIProxy().getRank();
	}

	boost::python::list evolveSingleStep(boost::python::list c) {
		boost::python::ssize_t len = boost::python::len(c);
		std::vector<double> activity = std::vector<double>();

		for(int i=0; i<len; i++) {
			double ca = boost::python::extract<double>(c[i]);
			activity.push_back(ca);
		}

		boost::python::list out;
		for(auto& it : network.evolveSingleStep(activity)) {
			out.append(it);
		}

		(*pb)++;

		return out;
	}

	void endSimulation() {
		network.endSimulation();
		t.stop();
		if(utilities::MPIProxy().getRank() == 0)
			t.report();
	}

protected:
	MPINetwork<Weight, NodeDistribution> network;
	report::handler::AbstractReportHandler *report_handler;
	boost::timer::auto_cpu_timer t;
	utilities::ProgressBar *pb;
	long _simulation_length; // ms
	double _time_step; // ms
	int _num_nodes;
};

template<class Weight, class NodeDistribution>
void define_python_MiindTvbModelAbstract()
{
	using namespace boost::python;
    class_<MiindTvbModelAbstract<Weight, NodeDistribution>>("MiindTvbModelAbstract", init<int,long>())
				.def("startSimulation", &MiindTvbModelAbstract<Weight, NodeDistribution>::startSimulation)
				.def("endSimulation", &MiindTvbModelAbstract<Weight, NodeDistribution>::endSimulation)
				.def("evolveSingleStep", &MiindTvbModelAbstract<Weight, NodeDistribution>::evolveSingleStep);
}

}

#endif // MPILIB_MIINDTVBMODELABSTRACT_HPP_
