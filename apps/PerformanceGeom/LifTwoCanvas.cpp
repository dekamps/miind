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

  LifNeuralDynamics dyn_e(PerformanceGeom::ODE_TWO_EXC,0.001);
  LifNeuralDynamics dyn_i(PerformanceGeom::ODE_TWO_INH,0.001);

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

  MPILib::CanvasParameter par_canvas;
  par_canvas._state_min     = -0.020;
  par_canvas._state_max     = 0.020;
  par_canvas._t_min         = 0.0;
  par_canvas._t_max         = 0.05;
  par_canvas._f_min         = 0.0;
  par_canvas._f_max         = 8.0;
  par_canvas._dense_min     = 0.0;
  par_canvas._dense_max     = 200.0;


  MPILib::report::handler::RootReportHandler handler("twopopcanvas.root", true, true, par_canvas);
  handler.addNodeToCanvas(id_excitatory_main);
  handler.addNodeToCanvas(id_inhibitory_main);

  SimulationRunParameter 
    par_run
    (
     handler,
     1000000000,
     0.,
     0.05,
     1e-4,
     1e-5,
     "twopop.log",
     1e-4
     );
   
  network.configureSimulation
  (
   par_run
  );

 
  network.evolve();
}
