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

	MiindTvbModelAbstract(int num_nodes, double simulation_length) :
		_num_nodes(num_nodes), _simulation_length(simulation_length){
	}

	~MiindTvbModelAbstract() {
	}

	virtual void init() {};

	virtual void startSimulation() {
		pb = new utilities::ProgressBar(network.startSimulation());
	}

	virtual std::vector<double> evolveSingleStep(std::vector<double> activity) {
		(*pb)++;
		return network.evolveSingleStep(activity);
	}

	virtual void endSimulation() {
		network.endSimulation();
		t.stop();
		t.report();
	}

	double getTimeStep() {
		return _time_step;
	}

	double getSimulationLength() {
		return _simulation_length;
	}

	int getNumNodes() {
		return _num_nodes;
	}

protected:
	MPINetwork<Weight, NodeDistribution> network;
	report::handler::AbstractReportHandler *report_handler;
	boost::timer::auto_cpu_timer t;
	utilities::ProgressBar *pb;
	double _simulation_length; // ms
	double _time_step; // ms
	int _num_nodes;
};

}

#endif // MPILIB_MIINDTVBMODELABSTRACT_HPP_
