#include <boost/python.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/PersistantAlgorithm.hpp>
#include <MPILib/include/DelayAlgorithmCode.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/WilsonCowanParameter.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>

typedef MPILib::MPINetwork<DelayedConnection, MPILib::utilities::CircularDistribution> Network;

class MiindModel {
private:
	Network network;
	boost::timer::auto_cpu_timer t;
	MPILib::utilities::ProgressBar *pb;
	long _simulation_length; // ms
	int _num_nodes;
	double _time_step; // ms
public:

	MiindModel(int num_nodes, long simulation_length, double dt) :
		_num_nodes(num_nodes), _simulation_length(simulation_length), _time_step(dt){
	}

	~MiindModel() {
		endSimulation();
	}

	void init()
	{

		try {
      if(_time_step != 0.000770095348827)
        std::invalid_argument( "Time Step must equal 0.000770095348827 to match the model file." );

      std::vector<std::string> vec_mat_0{"lif_0.005_0_0.mat"};
    	TwoDLib::MeshAlgorithm<DelayedConnection> alg_mesh_0("lif.model",vec_mat_0,_time_step);

      DelayedConnection con_external(6000,0.005,0);
			for(int i=0; i<_num_nodes; i++) {
				// generating nodes
				MPILib::NodeId id_0 = network.addNode(alg_mesh_0, MPILib::EXCITATORY_DIRECT);
        network.setNodeExternalSuccessor(id_0);
				network.setNodeExternalPrecursor(id_0, con_external);
			}

			std::string sim_name = "miind_wc";
			MPILib::report::handler::InactiveReportHandler handler =
											MPILib::report::handler::InactiveReportHandler();

			SimulationRunParameter par_run( handler,(_simulation_length/_time_step)+1,0,
											_simulation_length,_time_step,_time_step,sim_name,_time_step);

			network.configureSimulation(par_run);

		} catch(std::exception& exc){
			std::cout << exc.what() << std::endl;
		}
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
		if(utilities::MPIProxy().getRank() == 0) {
			t.report();
		}
	}
};

BOOST_PYTHON_MODULE(libmiindpython)
{
	using namespace boost::python;
	class_<MiindModel>("MiindModel", init<int,long,double>())
		.def("init", &MiindModel::init)
		.def("startSimulation", &MiindModel::startSimulation)
		.def("endSimulation", &MiindModel::endSimulation)
		.def("evolveSingleStep", &MiindModel::evolveSingleStep);
}
