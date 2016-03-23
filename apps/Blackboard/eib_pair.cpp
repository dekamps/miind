#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/algorithm/RateAlgorithmCode.hpp>
#include <MPILib/include/algorithm/BoxcarAlgorithmCode.hpp>
#include <MPILib/include/algorithm/DelayAssemblyAlgorithmCode.hpp>
#include <MPILib/include/algorithm/DelayAssemblyParameter.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/CsvReportHandler.hpp>

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
using MPILib::algorithm::RateAlgorithm;
using MPILib::algorithm::BoxcarAlgorithm;
using MPILib::algorithm::DelayAssemblyAlgorithm;
using MPILib::algorithm::DelayAssemblyParameter;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;
typedef MPILib::report::handler::CsvReportHandler Report;
typedef MPILib::algorithm::BoxcarAlgorithm<MPILib::DelayedConnection> BoxcarAlg;
typedef MPILib::algorithm::DelayAssemblyAlgorithm<MPILib::DelayedConnection> DecayAlg;

class EIB_Population_Pair
{
    public:
        NodeId inhibitory_node;
        NodeId excitatory_node;
        NodeId baseline_node;

        EIB_Population_Pair(){};

        EIB_Population_Pair(Network& network, Report& handler);
};

EIB_Population_Pair::EIB_Population_Pair(Network& network, Report& handler)
{
//Population density method alg
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
        par_ode1
        (
            n_bins,
            V_min,
            par_neuron,
            InitialDensityParameter(0.0, 0.0)
        );

    double min_bin = 0.01;
    LifNeuralDynamics dyn1(par_ode1,min_bin);
    LeakingOdeSystem sys1(dyn1);
    GeomParameter par_geom1(sys1);
    GeomDelayAlg alg1(par_geom1);

    OdeParameter
        par_ode2
        (
            n_bins,
            V_min,
            par_neuron,
            InitialDensityParameter(0.0, 0.0)
        );

    LifNeuralDynamics dyn2(par_ode2,min_bin);
    LeakingOdeSystem sys2(dyn2);
    GeomParameter par_geom2(sys2);
    GeomDelayAlg alg2(par_geom2);

//Fixed rate alg
    Rate rate_ext = 800.0;
    RateAlgorithm<MPILib::DelayedConnection> alg_ext(rate_ext);

// Configure nodes
    inhibitory_node = network.addNode(alg2, INHIBITORY_DIRECT);
    excitatory_node = network.addNode(alg1, EXCITATORY_DIRECT);
    baseline_node = network.addNode(alg_ext, EXCITATORY_DIRECT);

// Configure network graph
    MPILib::DelayedConnection con(50, 0.03, 0.0);
    MPILib::DelayedConnection icon(50, -0.04, 0.0);
    network.makeFirstInputOfSecond(excitatory_node, inhibitory_node, con);
    network.makeFirstInputOfSecond(inhibitory_node, excitatory_node, icon);
    network.makeFirstInputOfSecond(baseline_node, excitatory_node, con);
    network.makeFirstInputOfSecond(baseline_node, inhibitory_node, con);
}


class Gating_circuit
{
    public:
        // Internal nodes
        NodeId gate_keeper;
        NodeId gate;
        // Externally provided nodes
        NodeId source_assembly;
        NodeId control;
        NodeId target_assembly;

        Gating_circuit(){};

        Gating_circuit(Network& network, Report& handler,
                       NodeId s_assembly,
                       NodeId t_assembly,
                       NodeId ctrl,
                       GeomDelayAlg alg);
};

Gating_circuit::Gating_circuit(Network& network, Report& handler,
                               NodeId s_assembly,
                               NodeId t_assembly,
                               NodeId ctrl,
                               GeomDelayAlg alg)
{
    cout << "Building gating circuit" << endl;
    source_assembly = s_assembly;
    control = ctrl;
    target_assembly = t_assembly;

// Configure nodes
    gate_keeper = network.addNode(alg, INHIBITORY_DIRECT);
    gate  = network.addNode(alg, EXCITATORY_DIRECT);

// Configure network graph
    MPILib::DelayedConnection con(50, 0.03, 0.0);
    MPILib::DelayedConnection icon(60, -0.04, 0.0);
    network.makeFirstInputOfSecond(source_assembly, gate, con);
    network.makeFirstInputOfSecond(source_assembly, gate_keeper, con);
    network.makeFirstInputOfSecond(gate_keeper, gate, icon);
    network.makeFirstInputOfSecond(control, gate_keeper, icon);
    network.makeFirstInputOfSecond(gate, target_assembly, con);

}

class Memory_circuit
{
    public:
        // Internal nodes
        Gating_circuit forward_circuit;
        Gating_circuit backward_circuit;
        NodeId delay_assembly;
        // Externally provided nodes
        NodeId source_assembly;
        NodeId target_assembly;

        Memory_circuit(){};

        Memory_circuit(Network& network, Report& handler,
                       NodeId s_assembly,
                       NodeId t_assembly,
                       GeomDelayAlg alg,
                       DecayAlg decayalg);
};

Memory_circuit::Memory_circuit(Network& network, Report& handler,
                               NodeId s_assembly,
                               NodeId t_assembly,
                               GeomDelayAlg alg,
                               DecayAlg decayalg)
{
    cout << "Building memory circuit" << endl;
    source_assembly = s_assembly;
    target_assembly = t_assembly;

// Configure nodes
    delay_assembly  = network.addNode(decayalg, INHIBITORY_DIRECT);

    forward_circuit = Gating_circuit(network, handler, source_assembly,
                                     target_assembly, delay_assembly, alg);
    backward_circuit = Gating_circuit(network, handler, target_assembly,
                                      source_assembly, delay_assembly, alg);

    MPILib::DelayedConnection con(1, 0.03, 0.0);
    network.makeFirstInputOfSecond(source_assembly, delay_assembly, con);
    network.makeFirstInputOfSecond(target_assembly, delay_assembly, con);

// Configure network graph
// All connections were setup by gating circuits

}

class Control_circuit
{
    public:
        // Internal nodes
        Gating_circuit forward_circuit;
        Gating_circuit backward_circuit;
        // Externally provided nodes
        NodeId source_assembly;
        NodeId target_assembly;
        NodeId forward_control;
        NodeId backward_control;

        Control_circuit(){};

        Control_circuit(Network& network, Report& handler,
                        NodeId s_assembly,
                        NodeId t_assembly,
                        NodeId f_ctrl,
                        NodeId b_ctrl,
                        GeomDelayAlg alg);
};

Control_circuit::Control_circuit(Network& network, Report& handler,
                                 NodeId s_assembly,
                                 NodeId t_assembly,
                                 NodeId f_ctrl,
                                 NodeId b_ctrl,
                                 GeomDelayAlg alg)
{
    cout << "Building control circuit" << endl;
    source_assembly = s_assembly;
    target_assembly = t_assembly;
    forward_control = f_ctrl;
    backward_control = b_ctrl;

// Configure nodes
    forward_circuit = Gating_circuit(network, handler, source_assembly,
                                     target_assembly, forward_control, alg);
    backward_circuit = Gating_circuit(network, handler, target_assembly,
                                      source_assembly, backward_control, alg);

// Configure network graph
// All connections were setup by gating circuits

}

class BBcell_circuit
{
    public:
        // Internal nodes
        // fc stands for forward circuit
        // fc1 and bc1 correspond to linking source main and sub assemblies
        Control_circuit cc1;
        Control_circuit cc2;
        Memory_circuit mc;
        NodeId source_sub_assembly;
        NodeId target_sub_assembly;
        // Externally provided nodes
        NodeId source_main_assembly;
        NodeId target_main_assembly;
        NodeId control_fc1;
        NodeId control_fc2;
        NodeId control_bc1;
        NodeId control_bc2;

        BBcell_circuit(){};

        BBcell_circuit(Network& network, Report& handler,
                       NodeId s_assembly,
                       NodeId t_assembly,
                       NodeId ctrlfc1,
                       NodeId ctrlfc2,
                       NodeId ctrlbc1,
                       NodeId ctrlbc2,
                       GeomDelayAlg alg,
                       DecayAlg decayalg,
                       std::ofstream& node_desc);
};

BBcell_circuit::BBcell_circuit(Network& network, Report& handler,
                               NodeId s_assembly,
                               NodeId t_assembly,
                               NodeId ctrlfc1,
                               NodeId ctrlfc2,
                               NodeId ctrlbc1,
                               NodeId ctrlbc2,
                               GeomDelayAlg alg,
                               DecayAlg decayalg,
                               std::ofstream& node_desc)
{
    cout << "Building blackboard cell circuit" << endl;
    source_main_assembly = s_assembly;
    target_main_assembly = t_assembly;
    control_fc1 = ctrlfc1;
    control_fc2 = ctrlfc2;
    control_bc1 = ctrlbc1;
    control_bc2 = ctrlbc2;

// Configure nodes
    source_sub_assembly  = network.addNode(alg, EXCITATORY_DIRECT);
    target_sub_assembly  = network.addNode(alg, EXCITATORY_DIRECT);
    // Setup control circuit
    cc1 = Control_circuit(network, handler, source_main_assembly,
                          source_sub_assembly, control_fc1, control_bc1, alg);
    // Setup memory circuit
    mc = Memory_circuit(network, handler, source_sub_assembly,
                        target_sub_assembly, alg, decayalg);
    // Setup control circuit
    cc2 = Control_circuit(network, handler, target_sub_assembly,
                          target_main_assembly, control_fc2, control_bc2, alg);

// Write node description on file
    node_desc << "source-sub-assembly," << source_sub_assembly << endl;
    node_desc << "target-sub-assembly," << target_sub_assembly << endl;
    node_desc << "cc1_forward_gate-keeper," << cc1.forward_circuit.gate_keeper << endl;
    node_desc << "cc1_forward_gate," << cc1.forward_circuit.gate << endl;
    node_desc << "cc1_backward_gate-keeper," << cc1.backward_circuit.gate_keeper << endl;
    node_desc << "cc1_backward_gate," << cc1.backward_circuit.gate << endl;
    node_desc << "mc_delay-assembly," << mc.delay_assembly << endl;
    node_desc << "mc_forward_gate-keeper," << mc.forward_circuit.gate_keeper << endl;
    node_desc << "mc_forward_gate," << mc.forward_circuit.gate << endl;
    node_desc << "mc_backward_gate-keeper," << mc.backward_circuit.gate_keeper << endl;
    node_desc << "mc_backward_gate," << mc.backward_circuit.gate << endl;
    node_desc << "cc2_forward_gate-keeper," << cc2.forward_circuit.gate_keeper << endl;
    node_desc << "cc2_forward_gate," << cc2.forward_circuit.gate << endl;
    node_desc << "cc2_backward_gate-keeper," << cc2.backward_circuit.gate_keeper << endl;
    node_desc << "cc2_backward_gate," << cc2.backward_circuit.gate << endl;
}

int main(){
  //! [preamble]
    cout << "Demonstrating Blackboard circuit cell" << endl;

    Network network;
    Report handler("circuits_test", true);
    std::ofstream node_desc;
    node_desc.open("node_desc.csv");

// Configure algorithms

    //Fixed rate alg
    Rate rate_ext = 0.0;
    RateAlgorithm<MPILib::DelayedConnection> alg_ext(rate_ext);

    //Decay assembly alg
    DelayAssemblyParameter decayparam(0.5, 1000.0, 45.0, -45.0);
    DecayAlg decayalg(decayparam);

    //Boxcar alg
    std::vector<Event> ctrlevents(2);
    ctrlevents[0].start = 0.1;
    ctrlevents[0].end = 0.2;
    ctrlevents[0].rate = 1000.0;
    BoxcarAlg ctrl_box(ctrlevents);

    std::vector<Event> inputevents(2);
    inputevents[0].start = 0.0;
    inputevents[0].end = 0.05;
    inputevents[0].rate = 2000.0;
    BoxcarAlg input_box(inputevents);

    //Population density method alg
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

    EIB_Population_Pair pair(network, handler);
    node_desc << "node_name," << "node_id" << endl;
    node_desc << "inh," << pair.inhibitory_node << endl;
    node_desc << "exc," << pair.excitatory_node << endl;
    node_desc << "base," << pair.baseline_node << endl;

// // Setup main assemblies and controls
//     NodeId input_source = network.addNode(input_box, EXCITATORY_DIRECT);
//     NodeId input_target = network.addNode(input_box, EXCITATORY_DIRECT);

//     NodeId source_mem  = network.addNode(decayalg, EXCITATORY_DIRECT);
//     NodeId target_mem  = network.addNode(alg_ext, EXCITATORY_DIRECT);

//     NodeId source_main_assembly  = network.addNode(alg, EXCITATORY_DIRECT);
//     NodeId target_main_assembly  = network.addNode(alg, EXCITATORY_DIRECT);
//     NodeId ctrl1  = network.addNode(ctrl_box, INHIBITORY_DIRECT);
//     NodeId ctrl2  = network.addNode(alg_ext, INHIBITORY_DIRECT);
//     NodeId ctrl3  = network.addNode(alg_ext, INHIBITORY_DIRECT);
//     NodeId ctrl4  = network.addNode(ctrl_box, INHIBITORY_DIRECT);

//     MPILib::DelayedConnection con(50, 0.03, 0.0);
//     network.makeFirstInputOfSecond(input_source, source_mem, con);
//     network.makeFirstInputOfSecond(input_target, target_mem, con);
//     network.makeFirstInputOfSecond(source_mem, source_main_assembly, con);
//     network.makeFirstInputOfSecond(target_mem, target_main_assembly, con);

// // Write node description on file
//     node_desc << "node_name," << "node_id" << endl;
//     node_desc << "input-source," << input_source << endl;
//     node_desc << "input-target," << input_target << endl;
//     node_desc << "source-mem," << source_mem << endl;
//     node_desc << "target-mem," << target_mem << endl;
//     node_desc << "source-main-assembly," << source_main_assembly << endl;
//     node_desc << "target-main-assembly," << target_main_assembly << endl;
//     node_desc << "control-fc1," << ctrl1 << endl;
//     node_desc << "control-fc2," << ctrl2 << endl;
//     node_desc << "control-bc1," << ctrl4 << endl;
//     node_desc << "control-bc2," << ctrl3 << endl;

// // Create circuit
//     BBcell_circuit circuit(network, handler, source_main_assembly,
//                            target_main_assembly,
//                            ctrl1, ctrl2, ctrl3, ctrl4,
//                            alg, decayalg,
//                            node_desc);

// Setup simulation
    const SimulationRunParameter
        par_run
        (
            handler,
            10000000,
            0.0,  // start time
            3.0,  // end time
            1e-3,  // report step
            1e-4,  // simulation step
            "circuits.log"
        );
    network.configureSimulation(par_run);
    cout << "Circuit setup done. Starting simulation..." << endl;

// Run simulation
    network.evolve();
    node_desc.close();
    return 0;
}