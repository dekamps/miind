#include <boost/python.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/RootHighThroughputHandler.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/PersistantAlgorithm.hpp>
#include <MPILib/include/DelayAlgorithmCode.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/WilsonCowanParameter.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>

typedef MPILib::MPINetwork<double, MPILib::utilities::CircularDistribution> Network;

MPILib::Rate RateFunction_P(MPILib::Time t){
	return 1.25;
}

MPILib::Rate RateFunction_Q(MPILib::Time t){
	return 0.0;
}

class Wrapped {
private:
	Network network;
	boost::timer::auto_cpu_timer t;
	MPILib::utilities::ProgressBar *pb;
public:

	~Wrapped() {
		endSimulation();
	}

	void init()
	{

		try {	// generating algorithms
			Time E_tau = 8; // (TVB : tau_e) population time constant
			Rate E_max_rate = 1.0; // (TVB : c_e) maximum rate reached by the sigmoid function (E_max_rate / e^-alpha(x+theta))
			double E_noise = 1.3; // (TVB : a_e or alpha_e) noise term modulates the input (a term in Wilson Cowan)
			double E_bias = 4.0; // (TVB : b_e or theta_e) bias term shifts the input (theta term in Wilson Cowan)
			double E_input = 0.0; // the sum of all initial input rates

			MPILib::WilsonCowanParameter E_param = MPILib::WilsonCowanParameter(E_tau, E_max_rate, E_noise, E_bias, E_input);
			MPILib::WilsonCowanAlgorithm E_alg(E_param);

			Time I_tau = 8; // (TVB : tau_i) population time constant
			Rate I_max_rate = 1.0; // (TVB : c_i) maximum rate reached by the sigmoid function (I_max_rate / e^-alpha(x+theta))
			double I_noise = 2.0; // (TVB : a_i or alpha_i) noise term modulates the input
			double I_bias = 3.7; // (TVB : b_i or theta_i) bias term shifts the input
			double I_input = 0.0; // the sum of all initial input rates

			MPILib::WilsonCowanParameter I_param = MPILib::WilsonCowanParameter(I_tau, I_max_rate, I_noise, I_bias, I_input);
			MPILib::WilsonCowanAlgorithm I_alg(I_param);

			MPILib::Rate RateFunction_P(MPILib::Time);
			MPILib::RateFunctor<double> rate_functor_p(RateFunction_P);

			MPILib::Rate RateFunction_Q(MPILib::Time);
			MPILib::RateFunctor<double> rate_functor_q(RateFunction_Q);

			double E_E_Weight = 16; // (TVB : c_1)
			double I_E_Weight = -12; // (TVB : -c_2)
			double E_I_Weight = 15; // (TVB : c_3)
			double I_I_Weight = -3; //(TVB : -c_4)
			double P_E_Weight = 1;
			double Q_I_Weight = 1;
			double External_E_Weight = 1;

			// generation simulation parameter
			std::string sim_name = "wc/wc";
			MPILib::report::handler::RootHighThroughputHandler handler(sim_name,true);


			for(int i=0; i<1; i++) {
				// generating nodes
				MPILib::NodeId id_E = network.addNode(E_alg, MPILib::EXCITATORY_DIRECT);
				MPILib::NodeId id_I = network.addNode(I_alg, MPILib::INHIBITORY_DIRECT);
				MPILib::NodeId id_P = network.addNode(rate_functor_p, MPILib::NEUTRAL);
				MPILib::NodeId id_Q = network.addNode(rate_functor_q, MPILib::NEUTRAL);

				network.makeFirstInputOfSecond(id_E,id_E,E_E_Weight);
				network.makeFirstInputOfSecond(id_I,id_E,I_E_Weight);
				network.makeFirstInputOfSecond(id_E,id_I,E_I_Weight);
				network.makeFirstInputOfSecond(id_I,id_I,I_I_Weight);
				network.makeFirstInputOfSecond(id_P,id_E,P_E_Weight);
				network.makeFirstInputOfSecond(id_Q,id_I,Q_I_Weight);

				// generating connections
				network.setNodeExternalSuccessor(id_E);
				network.setNodeExternalPrecursor(id_E, External_E_Weight);
			}

			SimulationRunParameter par_run( handler,1000000,0,50.0,0.01,0.01,sim_name,0.01);
			network.configureSimulation(par_run);
			//network.evolve();
		} catch(std::exception& exc){
			//std::cout << exc.what() << std::endl;
		}
	}

	void evolve() {
		network.evolve();
	}

	void startSimulation() {
		pb = new MPILib::utilities::ProgressBar(network.startSimulation());
	}

	boost::python::list evolveSingleStep(boost::python::list c) {
		boost::python::ssize_t len = boost::python::len(c);
		std::vector<double> activity = std::vector<double>();

		for(int i=0; i<len; i++) {
			double ca = boost::python::extract<double>(c[i]);
			activity.push_back(ca);
		}

		network.setExternalPrecursorActivities(activity);

		network.evolveSingleStep();

		(*pb)++;

		boost::python::list out;
		for(auto& it : network.getExternalActivities()) {
			out.append(it);
		}

		return out;
	}

	void endSimulation() {
		network.endSimulation();
		t.stop();
		t.report();
	}
};

BOOST_PYTHON_MODULE(libmiindpw)
{
	using namespace boost::python;
	class_<Wrapped>("Wrapped")
		.def("init", &Wrapped::init)
		.def("evolve", &Wrapped::evolve)
		.def("startSimulation", &Wrapped::startSimulation)
		.def("endSimulation", &Wrapped::endSimulation)
		.def("evolveSingleStep", &Wrapped::evolveSingleStep);
}
