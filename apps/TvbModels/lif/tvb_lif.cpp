#include <boost/python.hpp>
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
#include <MPILib/include/MiindTvbModelAbstract.hpp>

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
class MiindModel : public MPILib::MiindTvbModelAbstract<DelayedConnection, MPILib::utilities::CircularDistribution> {
public:

	MiindModel(int num_nodes, long simulation_length) :
		MiindTvbModelAbstract(num_nodes, simulation_length){}

	void init(boost::python::list params)
	{
		_time_step = 0.000770095348827;
		for(int i=0; i<_num_nodes; i++) {
			std::vector<std::string> vec_mat_0{"lif_0.005_0_0.mat"};
    	TwoDLib::MeshAlgorithm<DelayedConnection> alg_mesh_0("lif.model",vec_mat_0,_time_step);

      DelayedConnection con_external(10000,0.005,0);
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
		report::handler::InactiveReportHandler handler = report::handler::InactiveReportHandler();

		SimulationRunParameter par_run( handler,(_simulation_length/_time_step)+1,0,
										_simulation_length,_time_step,_time_step,sim_name,_time_step);

		network.configureSimulation(par_run);
	}
};

// The module name (libmiindlif) must match the name of the generated shared
// library as defined in the CMakeLists file (libmiindlif.so)
BOOST_PYTHON_MODULE(libmiindlif)
{
	using namespace boost::python;
	define_python_MiindTvbModelAbstract<DelayedConnection, MPILib::utilities::CircularDistribution>();

	class_<MiindModel, bases<MPILib::MiindTvbModelAbstract<DelayedConnection,
				MPILib::utilities::CircularDistribution>>>("MiindModel", init<int,long>())
		.def("init", &MiindModel::init);
}
