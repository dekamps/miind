#include <boost/python.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/WilsonCowanParameter.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>

typedef MPILib::MPINetwork<double, MPILib::utilities::CircularDistribution> Network;

MPILib::Rate External_RateFunction(MPILib::Time t){
	return 1.0;
}
/* This class is designed to be used in conjunction with
 * tvb.simulator.models.Miind_WilsonCowan.py only. Any updates to this shared library
 * should be communicated to the TVB team and a new .so file should be provided
 * to be included in the tvb-library source tree (tvb-library/tvb/simulator/models/).
 *
 * Note that MPI_init is not called (through boost::mpi::environment or otherwise)
 * It is expected that, while TVB source doesn't need to know about MPI, the calling
 * working directory script does (otherwise, why are you calling eg mpiexec/mpirun?)
 */
class MiindWilsonCowan {
private:
	Network network;
	boost::timer::auto_cpu_timer t;
	MPILib::utilities::ProgressBar *pb;
	std::vector<double> E_Initials = std::vector<double> (); // initial excitatory values
	std::vector<double> I_Initials = std::vector<double> (); // initial inhibitory values
	long _simulation_length; // ms
	int _num_nodes;
	double _time_step; // ms
public:

	MiindWilsonCowan(int num_nodes, long simulation_length, double dt) :
		_num_nodes(num_nodes), _simulation_length(simulation_length), _time_step(dt){
	}

	~MiindWilsonCowan() {
		endSimulation();
	}

	// In general, we won't need to set initial values but we implement it here
	// so that we can match TVB's Wilson Cowan example.
	void setInitialValues(boost::python::list E_vals, boost::python::list I_vals) {
		for(int i=0; i<_num_nodes; i++) {
			E_Initials.push_back(boost::python::extract<double>(E_vals[i]));
			I_Initials.push_back(boost::python::extract<double>(I_vals[i]));
		}
	}

	void init(boost::python::list params)
	{
		// (TVB : tau_e) population time constant
		Time   E_tau       = boost::python::extract<double>(params[4]);
		// (TVB : c_e) maximum rate reached by the sigmoid function
		Rate   E_max_rate  = boost::python::extract<double>(params[6]);
		// (TVB : a_e) noise term modulates the input (a term in Wilson Cowan)
		double E_noise     = boost::python::extract<double>(params[10]);
		// (TVB : b_e) bias term shifts the input (theta term in Wilson Cowan)
		double E_bias      = boost::python::extract<double>(params[12]);
		// (TVB : r_e) smoothing term over output which models refractory dynamics
		double E_smoothing = boost::python::extract<double>(params[8]);
		// (TVB : tau_i) population time constant
		Time   I_tau       = boost::python::extract<double>(params[5]);
		// (TVB : c_i) maximum rate reached by the sigmoid function
		Rate   I_max_rate  = boost::python::extract<double>(params[7]);
		// (TVB : a_i) noise term modulates the input
		double I_noise     = boost::python::extract<double>(params[11]);
		// (TVB : b_i) bias term shifts the input
		double I_bias      = boost::python::extract<double>(params[13]);
		// (TVB : r_e) smoothing term over output which models refractory dynamics
		double I_smoothing = boost::python::extract<double>(params[9]);
		// (TVB : c_ee) Weight of E->E connection
		double E_E_Weight  = boost::python::extract<double>(params[0]);
		// (TVB : -c_ie) Weight of E->I connection
		double I_E_Weight  = -boost::python::extract<double>(params[1]);
		 // (TVB : c_ei) Weight of I->E connection
		double E_I_Weight  = boost::python::extract<double>(params[2]);
		// (TVB : -c_ii) Weight of I->I connection
		double I_I_Weight  = -boost::python::extract<double>(params[3]);
		// (TVB : P) Some additional drive to excitatory pop
		double P_E_Weight  = boost::python::extract<double>(params[14]);
		// (TVB : Q) Some additional drive to inhibitory pop
		double Q_I_Weight  = boost::python::extract<double>(params[15]);

		try {
			std::vector<NodeId> E_ids = std::vector<NodeId>();
			std::vector<NodeId> I_ids = std::vector<NodeId>();

			for(int i=0; i<_num_nodes; i++) {
				MPILib::WilsonCowanParameter E_param = MPILib::WilsonCowanParameter(
								E_tau, E_max_rate, E_noise, E_bias, E_Initials[i], E_smoothing);
				MPILib::WilsonCowanAlgorithm E_alg(E_param);

				MPILib::WilsonCowanParameter I_param = MPILib::WilsonCowanParameter(
								I_tau, I_max_rate, I_noise, I_bias, I_Initials[i], I_smoothing);
				MPILib::WilsonCowanAlgorithm I_alg(I_param);

				MPILib::Rate RateFunction_P(MPILib::Time);
				MPILib::RateFunctor<double> rate_functor_p(External_RateFunction);

				MPILib::Rate RateFunction_Q(MPILib::Time);
				MPILib::RateFunctor<double> rate_functor_q(External_RateFunction);

				MPILib::NodeId id_E = network.addNode(E_alg, MPILib::EXCITATORY_DIRECT);
				MPILib::NodeId id_I = network.addNode(I_alg, MPILib::INHIBITORY_DIRECT);
				MPILib::NodeId id_P = network.addNode(rate_functor_p, MPILib::NEUTRAL);
				MPILib::NodeId id_Q = network.addNode(rate_functor_q, MPILib::NEUTRAL);

				E_ids.push_back(id_E);
				I_ids.push_back(id_I);

				network.makeFirstInputOfSecond(id_E,id_E,E_E_Weight);
				network.makeFirstInputOfSecond(id_I,id_E,I_E_Weight);
				network.makeFirstInputOfSecond(id_E,id_I,E_I_Weight);
				network.makeFirstInputOfSecond(id_I,id_I,I_I_Weight);
				network.makeFirstInputOfSecond(id_P,id_E,P_E_Weight);
				network.makeFirstInputOfSecond(id_Q,id_I,Q_I_Weight);

			}

			// Set each node to have an external successor (for coupling input from TVB)
			for(auto& id : E_ids) {
				network.setNodeExternalSuccessor(id);
				network.setNodeExternalPrecursor(id, 1);
			}

			for(auto& id : I_ids) {
				network.setNodeExternalSuccessor(id);
				network.setNodeExternalPrecursor(id, 1);
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

	int startSimulation() {
		pb = new MPILib::utilities::ProgressBar(network.startSimulation());

		// child processes just loop here - evolve isn't called here to avoid a deadlock
		// issue - would like to fix.
		if(utilities::MPIProxy().getRank() > 0)
			for(int i=0; i<int(_simulation_length/_time_step); i++)
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
		if(utilities::MPIProxy().getRank() == 0) {
			t.report();
		}
	}
};

// The module name (libmiindwc) must match the name of the generated shared
// library as defined in the CMakeLists file (libmiindwc.so)
BOOST_PYTHON_MODULE(libmiindwc)
{
	using namespace boost::python;
	class_<MiindWilsonCowan>("MiindWilsonCowan", init<int,long,double>())
		.def("setInitialValues", &MiindWilsonCowan::setInitialValues)
		.def("init", &MiindWilsonCowan::init)
		.def("startSimulation", &MiindWilsonCowan::startSimulation)
		.def("endSimulation", &MiindWilsonCowan::endSimulation)
		.def("evolveSingleStep", &MiindWilsonCowan::evolveSingleStep);
}
