#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/RootReportHandler.hpp>
#include "CreateTwoPopulationNetworkCode.hpp"
#include "TwoPopulationDefinitions.hpp"

using GeomLib::GeomAlgorithm;
using GeomLib::GeomParameter;
using GeomLib::InitialDensityParameter;
using GeomLib::LeakingOdeSystem;
using GeomLib::LifNeuralDynamics;
using GeomLib::OdeParameter;
using GeomLib::NeuronParameter;

using MPILib::EXCITATORY_DIRECT;
using MPILib::NodeId;
using MPILib::SimulationRunParameter;
using MPILib::RateAlgorithm;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;

int main(){

  InitialDensityParameter dense(0.0,0.0);
  double lambda = 0.001;
  LifNeuralDynamics dyn_e(PerformanceGeom::ODE_TWO_EXC,lambda);
   std::cout << "Time step for the excitatory grid: " << dyn_e.TStep() <<  std::endl;
   std::cout << "This is then the time step for the network!" << std::endl;
   Time tau_i = PerformanceGeom::ODE_TWO_INH._par_pop._tau;
   std::cout << "The membrane time constant for the inhibitory population: " << tau_i <<  std::endl;
   Number N_pos_i = PerformanceGeom::ODE_TWO_INH._nr_bins;
   std::cout << "The number of bins for the positive range: " << N_pos_i << std::endl;
   OdeParameter ode_new_i = PerformanceGeom::ODE_TWO_INH;
   double fn_bins = 1 - (tau_i/dyn_e.TStep())*log(lambda);
   std::cout << "New number of bins: " << fn_bins << std::endl;
   ode_new_i._nr_bins = fn_bins;
   double lambda_i  = exp( -double(ode_new_i._nr_bins - 1)*dyn_e.TStep()/tau_i);
   std::cout << "We have dt in the right ball park, but N_i is an integer, so let's adapt "
		     << "lambda to get the time step exactly right: " << lambda_i << std::endl;
   LifNeuralDynamics dyn_i(ode_new_i,lambda_i);
   std::cout << "The new time step is indeed: " << dyn_i.TStep() << std::endl;

  LeakingOdeSystem sys_e(dyn_e);
  LeakingOdeSystem sys_i(dyn_i);

  GeomParameter par_e(sys_e,"LifNumericalMasterEquation",PerformanceGeom::EXC_DIFF);
  GeomParameter par_i(sys_i,"LifNumericalMasterEquation",PerformanceGeom::INH_DIFF);

  NodeId id_cortical_background;
  NodeId id_excitatory_main;
  NodeId id_inhibitory_main;
  NodeId id_rate;
  Network network =
    PerformanceGeom::CreateTwoPopulationNetwork<GeomAlgorithm<MPILib::DelayedConnection>  >
    (
     &id_cortical_background,
     &id_excitatory_main,
     &id_inhibitory_main,
     &id_rate,
     par_e,
     par_i
     );


  MPILib::report::handler::RootReportHandler handler("twopopcanvas.root", false);
  handler.addNodeToCanvas(id_excitatory_main);
  handler.addNodeToCanvas(id_inhibitory_main);

  SimulationRunParameter
    par_run
    (
     handler,
     1000000000,
     0.,
     0.05,
     1e-3,
     dyn_e.TStep(),
     "twopop.log",
     1e-3
     );

 
  network.configureSimulation
  (
   par_run
   );

 
  network.evolve();
}
