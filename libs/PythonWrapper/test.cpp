#include <boost/python.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/MinimalReportHandler.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/PersistantAlgorithm.hpp>
#include <MPILib/include/DelayAlgorithmCode.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;

class Wrapped {
private:
	Network network;
public:

	void init()
	{
		try {	// generating algorithms

			std::vector<std::string> vec_mat_0{"lif_0.005_0_0.mat"};
			TwoDLib::MeshAlgorithm<DelayedConnection> alg_mesh_0("lif.model",vec_mat_0,0.000770095348827);

			DelayedConnection con_1_0_0(3,0.005,0);

			for(int i=0; i<76; i++) {
				// generating nodes
				MPILib::NodeId id_0 = network.addNode(alg_mesh_0,MPILib::EXCITATORY_DIRECT);

				// generating connections
				network.defineExternalNodeInputAndOutput(id_0,id_0,con_1_0_0,con_1_0_0);
			}

			// generation simulation parameter
			std::string sim_name = "lif/lif_";
			MPILib::report::handler::MinimalReportHandler handler(sim_name,true);

			SimulationRunParameter par_run( handler,1000000,0,5.0,0.00770095348827,0.000770095348827,sim_name,0.00770095348827);
			network.configureSimulation(par_run);
			//network.evolve();
		} catch(std::exception& exc){
			//std::cout << exc.what() << std::endl;
		}
	}

	void startSimulation() {
		network.startSimulation();
	}

	boost::python::list evolveSingleStep(boost::python::list c) {
		boost::python::ssize_t len = boost::python::len(c);
		std::vector<double> activity = std::vector<double>();

		for(int i=0; i<len; i++) {
			double ca = boost::python::extract<double>(c[i]);
			activity.push_back((ca*2000));
		}

		network.setExternalActivities(activity);

		network.evolveSingleStep();

		boost::python::list out;
		for(auto& it : network.getExternalActivities()) {
			out.append(it);
		}

		return out;
	}

	void endSimulation() {
		network.endSimulation();
	}
};

BOOST_PYTHON_MODULE(libmiindpw)
{
	using namespace boost::python;
	class_<Wrapped>("Wrapped")
		.def("init", &Wrapped::init)
		.def("startSimulation", &Wrapped::startSimulation)
		.def("endSimulation", &Wrapped::endSimulation)
		.def("evolveSingleStep", &Wrapped::evolveSingleStep);
}
