#include <boost/python.hpp>

#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/RootReportHandler.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/PersistantAlgorithm.hpp>
#include <MPILib/include/DelayAlgorithmCode.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;

MPILib::Rate RateFunction_1(MPILib::Time t){
	return 0.;
}

class Wrapped {
private:
	Network network;
	std::vector<MPILib::MPINode<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution>*> nodes =
		std::vector<MPILib::MPINode<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution>*>();
public:

	void addNode() {
		try {	// generating algorithms
			std::vector<std::string> vec_mat_0{"lif_0.005_0_0.mat"};
			TwoDLib::MeshAlgorithm<DelayedConnection> alg_mesh_0("lif.model",vec_mat_0,0.000770095348827);
			MPILib::Rate RateFunction_1(MPILib::Time);
			MPILib::RateFunctor<DelayedConnection> rate_functor_1(RateFunction_1);
			// generating nodes
			MPILib::NodeId id_0 = network.addNode(alg_mesh_0,MPILib::EXCITATORY_DIRECT);
			MPILib::NodeId id_1 = network.addNode(rate_functor_1,MPILib::NEUTRAL);
			nodes.push_back(network.getNode(id_0));
			// generating connections
			DelayedConnection con_1_0_0(3,0.005,0);
			network.makeFirstInputOfSecond(id_1,id_0,con_1_0_0);
		} catch(std::exception& exc){
			//std::cout << exc.what() << std::endl;
		}
	}

	void init()
	{
		try {	// generating algorithms
			// generation simulation parameter
			std::string sim_name = "lif/lif_";
			MPILib::report::handler::RootReportHandler handler(sim_name,true);

			SimulationRunParameter par_run( handler,1000000,0,5.0,0.00770095348827,0.000770095348827,sim_name,0.00770095348827);
			network.configureSimulation(par_run);
			//network.evolve();
		} catch(std::exception& exc){
			//std::cout << exc.what() << std::endl;
		}
	}

	void setPrecurserActivity(boost::python::list c) {
		boost::python::ssize_t len = boost::python::len(c);

		for(int i=0; i<len; i++) {
			double ca = boost::python::extract<double>(c[i]);

			std::vector<double> activity = std::vector<double>();
			activity.push_back((ca*2000));
			nodes[i]->setPrecurserActivity(activity);
		}

	}

	boost::python::list evolveSingleStep() {
		for(int i=0; i<nodes.size(); i++) {
			std::vector<ActivityType> activity = nodes[i]->getPrecurserActivity();
			printf("Coupled %f\n", activity[1]);
		}

		network.evolveSingleStep();

		boost::python::list out;
		for(int i=0; i<nodes.size(); i++) {
			printf("Activity %f\n", nodes[i]->getActivity());
			out.append(nodes[i]->getActivity());
		}

		return out;
	}
};

BOOST_PYTHON_MODULE(libmiindpw)
{
	using namespace boost::python;
	class_<Wrapped>("Wrapped")
		.def("init", &Wrapped::init)
		.def("addNode", &Wrapped::addNode)
		.def("setPrecurserActivity", &Wrapped::setPrecurserActivity)
		.def("evolveSingleStep", &Wrapped::evolveSingleStep);
}
