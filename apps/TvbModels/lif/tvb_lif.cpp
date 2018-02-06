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
#include <MPILib/include/DelayAlgorithmCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

typedef MPILib::MPINetwork<DelayedConnection, MPILib::utilities::CircularDistribution> Network;

/** This class is generically named MiindModel so no code change is required in TVB.
 * We can implement any simulation we desire and once the shared library is generated,
 * it can be copied to the TVB working directory and referenced when
 * instantiating the tvb.simulator.models.Miind class.
 *
 * e.g tvb.simulator.models.Miind('libmiindlif.so',76, 1**2, 0.000770095348827)
 * Note that MPI_init is not called (through boost::mpi::environment or otherwise)
 * It is expected that, while TVB source doesn't need to know about MPI, the calling
 * working directory script does (otherwise, why are you calling eg mpiexec/mpirun?)
 */
class MiindModel {
private:
	Network network;
	boost::timer::auto_cpu_timer t;
	MPILib::utilities::ProgressBar *pb;
	long _simulation_length; // ms
	int _num_nodes;
public:

	MiindModel(int num_nodes, long simulation_length) :
		_num_nodes(num_nodes), _simulation_length(simulation_length){
	}

	~MiindModel() {
		endSimulation();
	}

	void init()
	{
		try {

			for(int i=0; i<_num_nodes; i++) {
				std::vector<std::string> vec_mat_0{"lif_0.005_0_0.mat"};
	    	TwoDLib::MeshAlgorithm<DelayedConnection> alg_mesh_0("lif.model",vec_mat_0,0.000770095348827);

	      DelayedConnection con_external(6000,0.005,0);
				// As a basic example, we just implement a single lif node for each region
				// in TVB's connectivity.
				MPILib::NodeId id_0 = network.addNode(alg_mesh_0, MPILib::EXCITATORY_DIRECT);
				// Each node must have an incoming and outgoing connection to TVB
        network.setNodeExternalSuccessor(id_0);
				network.setNodeExternalPrecursor(id_0, con_external);
			}


			/* Any handler is permitted but as TVB provides it's own output, we get a speed
			 * increase by doing no IO in MIIND.
			 * If ROOT handlers are used, remember that the sim_name must include a
			 * directory and that directory must exist in the working directory.
			 * e.g sim_name = miind_output/miind_lif
			 */
			std::string sim_name = "miind_lif";
			MPILib::report::handler::InactiveReportHandler handler =
											MPILib::report::handler::InactiveReportHandler();

			SimulationRunParameter par_run( handler,(_simulation_length/0.000770095348827)+1,0,
											_simulation_length,0.000770095348827,0.000770095348827,sim_name,0.000770095348827);

			network.configureSimulation(par_run);

		} catch(std::exception& exc){
			std::cout << exc.what() << std::endl;
		}
	}

	int startSimulation() {
		pb = new MPILib::utilities::ProgressBar(network.startSimulation());

		// child processes just loop here - evolve isn't called here to avoid a deadlock
		// issue - would like to fix.

		if(utilities::MPIProxy().getRank() > 0)
			for(int i=0; i<int(_simulation_length/0.000770095348827)+1; i++)
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
};

// The module name (libmiindlif) must match the name of the generated shared
// library as defined in the CMakeLists file (libmiindlif.so)
BOOST_PYTHON_MODULE(libmiindlif)
{
	using namespace boost::python;
	class_<MiindModel>("MiindModel", init<int,long>())
		.def("init", &MiindModel::init)
		.def("startSimulation", &MiindModel::startSimulation)
		.def("endSimulation", &MiindModel::endSimulation)
		.def("evolveSingleStep", &MiindModel::evolveSingleStep);
}
