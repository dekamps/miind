#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/algorithm/RateAlgorithmCode.hpp>
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
using MPILib::algorithm::RateAlgorithm;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;

int main(){
  InitialDensityParameter dense(0.0,0.0);

  LifNeuralDynamics dyn_e(PerformanceGeom::ODE_TWO_EXC,0.001);
  LifNeuralDynamics dyn_i(PerformanceGeom::ODE_TWO_INH,0.001);

  LeakingOdeSystem sys_e(dyn_e);
  LeakingOdeSystem sys_i(dyn_i);

  GeomParameter par_e(sys_e,1.0,PerformanceGeom::EXC_DIFF);
  GeomParameter par_i(sys_i,1.0,PerformanceGeom::INH_DIFF);

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

 
  network.configureSimulation
  (
   PerformanceGeom::TWOPOP_PARAMETER
   );

 
  network.evolve();
}
