//Machine-generated by miind.py. Edit at your own risk.

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
typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
	// defining variables
	const double omega = 1.0;
int main(int argc, char *argv[]){
	Network network;
	boost::timer::auto_cpu_timer t;

#ifdef ENABLE_MPI
	// initialise the mpi environment this cannot be forwarded to a class
	boost::mpi::environment env(argc, argv);
#endif

	try {	// generating algorithms
	std::vector<std::string> vec_mat_0{"aexp97b7e1e7-b88c-4ff8-8e86-7e1e092f4d9a_-1_0_0_0_.mat","aexp97b7e1e7-b88c-4ff8-8e86-7e1e092f4d9a_1_0_0_0_.mat"};
	TwoDLib::MeshAlgorithm<DelayedConnection,TwoDLib::MasterOdeint> alg_mesh_0("aexp97b7e1e7-b88c-4ff8-8e86-7e1e092f4d9a.model",vec_mat_0,0.00002, 0.0);
	MPILib::Rate RateFunction_1(MPILib::Time);
	MPILib::RateFunctor<DelayedConnection> rate_functor_1(RateFunction_1);
	MPILib::Rate RateFunction_2(MPILib::Time);
	MPILib::RateFunctor<DelayedConnection> rate_functor_2(RateFunction_2);
	// generating nodes
	MPILib::NodeId id_0 = network.addNode(alg_mesh_0,MPILib::EXCITATORY_DIRECT);
	MPILib::NodeId id_1 = network.addNode(rate_functor_1,MPILib::NEUTRAL);
	MPILib::NodeId id_2 = network.addNode(rate_functor_2,MPILib::NEUTRAL);
	// generating connections
	DelayedConnection con_1_0_0(1,1.,0);
	network.makeFirstInputOfSecond(id_1,id_0,con_1_0_0);
	DelayedConnection con_2_0_0(1,-1.,0);
	network.makeFirstInputOfSecond(id_2,id_0,con_2_0_0);
	// generation simulation parameter
	MPILib::report::handler::RootReportHandler handler("single",true);

	SimulationRunParameter par_run( handler,1000000,0,0.4,1e-03,0.0001,"single.log",1e-03);
	network.configureSimulation(par_run);
	network.evolve();
	} catch(std::exception& exc){
		std::cout << exc.what() << std::endl;
#ifdef ENABLE_MPI
	//Abort the MPI environment in the correct way :
	env.abort(1);
#endif
	}

	MPILib::utilities::MPIProxy().barrier();
	t.stop();
	if (MPILib::utilities::MPIProxy().getRank() == 0) {

		std::cout << "Overall time spend\n";
		t.report();
	}
	return 0;
}
MPILib::Rate RateFunction_1(MPILib::Time t){
	return 3000.;
}
MPILib::Rate RateFunction_2(MPILib::Time t){
	return t < 0.1 ? 1000. : 0.;
}
