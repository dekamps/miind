#include <GeomLib.hpp>
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/BoxcarAlgorithmCode.hpp>
#include <MPILib/include/DelayAssemblyAlgorithmCode.hpp>
#include <MPILib/include/DelayAssemblyParameter.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/CsvReportHandler.hpp>
#include <MPILib/include/AlgorithmInterface.hpp>

using GeomLib::GeomAlgorithm;
using GeomLib::GeomParameter;
using GeomLib::InitialDensityParameter;
using GeomLib::LeakingOdeSystem;
using GeomLib::LifNeuralDynamics;
using GeomLib::OdeParameter;
using GeomLib::NeuronParameter;

using MPILib::EXCITATORY_DIRECT;
using MPILib::INHIBITORY_DIRECT;
using MPILib::NodeId;
using MPILib::Event;
using MPILib::SimulationRunParameter;
using MPILib::RateAlgorithm;
using MPILib::BoxcarAlgorithm;
using MPILib::DelayAssemblyAlgorithm;
using MPILib::DelayAssemblyParameter;

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::getline;
using std::stoi;
using std::stof;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;
typedef MPILib::report::handler::CsvReportHandler Report;
typedef MPILib::BoxcarAlgorithm<MPILib::DelayedConnection> BoxcarAlg;
typedef MPILib::DelayAssemblyAlgorithm<MPILib::DelayedConnection> DelayAlg;
typedef MPILib::AlgorithmInterface<MPILib::DelayedConnection> Algorithm;

/**
 * @brief EIB_Population_Pair
 *
 * Contains mutually connected excitatory and inhibitory nodes.
 * Both nodes receive activity from a baseline node. The idea is to instantiate
 * a fix baseline rate feeding into a balanced activity neural population.
 */
class Connectivity_Structure
{
    public:
        double efficacy;
        int baseline_connections;
        int forward_connections;
        int inhibitory_connections;
        double baseline_slope;

        Connectivity_Structure(){};

        Connectivity_Structure(double eff, int bcon, int fcon, int icon, double bslope);
};

Connectivity_Structure::Connectivity_Structure(double eff, int bcon, int fcon,
                                               int icon, double bslope)
{
    efficacy = eff;
    baseline_connections = bcon;
    forward_connections = fcon;
    inhibitory_connections = icon;
    baseline_slope = bslope;
}

/**
 * @brief EIB_Population_Pair
 *
 * Contains mutually connected excitatory and inhibitory nodes.
 * Both nodes receive activity from a baseline node. The idea is to instantiate
 * a fix baseline rate feeding into a balanced activity neural population.
 */
class EIB_Population_Pair
{
    public:
        NodeId inhibitory_node;
        NodeId excitatory_node;
        NodeId baseline_node;
        NodeId aha_node;
        int is_excitatory;

        EIB_Population_Pair(){};

        EIB_Population_Pair(Network& network, Report& handler, Rate baseline,
                            int Exc, Algorithm& alg, Connectivity_Structure con_struct);
};

EIB_Population_Pair::EIB_Population_Pair(Network& network, Report& handler,
                                         Rate baseline, int Exc, Algorithm& alg,
                                         Connectivity_Structure con_struct)
{
    is_excitatory = Exc;
    // Delay assembly algorithm for baseline node
    // Works as a fixed rate node with a slope on initialization
    DelayAssemblyParameter decayparam(10.0, baseline, -1.0, -10.0, con_struct.baseline_slope);  // was 500, 80
    DelayAlg delayalg(decayparam);

    // Configure nodes
    baseline_node  = network.addNode(delayalg, EXCITATORY_DIRECT);

    // Configure network graph
    // The graph allows to transform the EI balanced network into a simple
    // forward activity pair of nodes to study the impact of EI balance.

    // EFFICACY AND CONNECTIONS
    MPILib::DelayedConnection bas(con_struct.baseline_connections,
                                  con_struct.efficacy, 0.0);

    if (is_excitatory == 1){
        excitatory_node = network.addNode(alg, EXCITATORY_DIRECT);
        network.makeFirstInputOfSecond(baseline_node, excitatory_node, bas);
    } else {
        inhibitory_node = network.addNode(alg, INHIBITORY_DIRECT);
        network.makeFirstInputOfSecond(baseline_node, inhibitory_node, bas);
    }
}

int main(int argc, char *argv[]){
    // configuration_file = argv[1]
    string configfile = argv[1];
    std::ifstream config(configfile);
    string str;

    // write node description
    getline(config, str);
    int write_desc = stoi(str);

    // configure circuit events
    getline(config, str);
    double baseline_rate = stof(str);
    getline(config, str);
    double baseline_slope = stof(str);

    // configuration network connections
    getline(config, str);
    double eff = stof(str);
    getline(config, str);
    int bcon = stoi(str);

    // simulation time in s (since starts at 0.)
    getline(config, str);
    double end_sim = stof(str);

//! [preamble]
    cout << "\nComputing dynamics of Blackboard compartment circuit" << endl;
    cout << "LIF model with parameters:" << endl;
    cout << "Configuration from: " << configfile << endl;

    // Initialize network and report file for activity simulation
    // Store activity as simple records with time, Nodeid and activity rate.
    Network network;
    Report handler(configfile, true);

// Configure algorithms

    // Population density method algorithm passed to circuit classes to
    // instantiate new nodes.

    Number    n_bins = 330;
    Potential V_min  = 0.00;

    NeuronParameter
        par_neuron
        (
            1.0,   // v_threshold
            0.0,   // v_reset
            0.0,   // v_reversal
            0.0,   // tau_refractive
            50e-3  // tau
        );

    OdeParameter
        par_ode
        (
            n_bins,
            V_min,
            par_neuron,
            InitialDensityParameter(0.0, 0.0)
        );

    double min_bin = 0.01;
    LifNeuralDynamics dyn(par_ode,min_bin);
    LeakingOdeSystem sys(dyn);
    GeomParameter par_geom(sys);
    GeomDelayAlg alg(par_geom);

//Setup connectivity structure
    Connectivity_Structure con_struct(eff, bcon, 0, 0, baseline_slope);

// Setup nodes
    EIB_Population_Pair pop(network, handler, baseline_rate, 1, alg, con_struct);

    // Generate a semantic node description file
    if(write_desc > 0){
        std::ofstream node_desc;
        node_desc.open("node_desc_simplepop.csv");

        // Write node description on file for all created nodes
        node_desc << "node_name," << "node_id," << "node_category" << endl;
        node_desc << "source-main-assembly," << pop.excitatory_node << ",main-assembly" << endl;
        node_desc << "source-main-assembly-b," << pop.baseline_node << ",baseline" << endl;

        node_desc.close();
    }
// Setup simulation
    const SimulationRunParameter
        par_run
        (
            handler,
            10000000,
            0.0,  // start time
            end_sim,  // end time
            1e-3,  // report step
            dyn.TStep(),  // simulation step
            "activity.log"
        );
    network.configureSimulation(par_run);
    cout << "Circuit setup done. Starting simulation..." << endl;

// Run simulation
    network.evolve();
    return 0;
}
