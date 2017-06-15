#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/BoxcarAlgorithmCode.hpp>
#include <MPILib/include/DelayAssemblyAlgorithmCode.hpp>
#include <MPILib/include/DelayAssemblyParameter.hpp>
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
using MPILib::RateAlgorithm;
using MPILib::BoxcarAlgorithm;
using MPILib::DelayAssemblyAlgorithm;
using MPILib::DelayAssemblyParameter;

using std::cout;
using std::endl;

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;
typedef GeomLib::GeomAlgorithm<MPILib::DelayedConnection> GeomDelayAlg;
typedef MPILib::report::handler::CsvReportHandler Report;
typedef MPILib::BoxcarAlgorithm<MPILib::DelayedConnection> BoxcarAlg;
typedef MPILib::DelayAssemblyAlgorithm<MPILib::DelayedConnection> DecayAlg;

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

        EIB_Population_Pair(){};

        EIB_Population_Pair(Network& network, Report& handler, Rate baseline);
};

EIB_Population_Pair::EIB_Population_Pair(Network& network, Report& handler,
                                         Rate baseline)
{
    // Population density method algorithm for Excitatory and Inhibitory nodes
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

    // Instantiate algorithm for excitatory node
    OdeParameter
        par_ode1
        (
            n_bins,
            V_min,
            par_neuron,
            InitialDensityParameter(0.0, 0.0)
        );

    double min_bin = 0.01;
    LifNeuralDynamics dyn1(par_ode1, min_bin);
    LeakingOdeSystem sys1(dyn1);
    GeomParameter par_geom1(sys1);
    GeomDelayAlg alg1(par_geom1);

    // Instantiate algorithm for inhibitory node
    OdeParameter
        par_ode2
        (
            n_bins,
            V_min,
            par_neuron,
            InitialDensityParameter(0.0, 0.0)
        );

    LifNeuralDynamics dyn2(par_ode2, min_bin);
    LeakingOdeSystem sys2(dyn2);
    GeomParameter par_geom2(sys2);
    GeomDelayAlg alg2(par_geom2);

    // Delay assembly algorithm for baseline node
    // Works as a fixed rate node with a slope on initialization
    DelayAssemblyParameter decayparam(10.0, baseline, -1.0, -10.0);  // was 500, 80
    DecayAlg decayalg(decayparam);

    // Configure nodes
    inhibitory_node = network.addNode(alg2, INHIBITORY_DIRECT);
    excitatory_node = network.addNode(alg1, EXCITATORY_DIRECT);
    baseline_node  = network.addNode(decayalg, EXCITATORY_DIRECT);

    // Configure network graph
    // The graph allows to transform the EI balanced network into a simple
    // forward activity pair of nodes to study the impact of EI balance.
    // We modify here to turn on and off EI mechanisms.
    // Off at the moment (0 efficacy in connections between EI nodes)
    MPILib::DelayedConnection con(1, 0.0, 0.0);
    network.makeFirstInputOfSecond(excitatory_node, inhibitory_node, con);
    MPILib::DelayedConnection icon(1, -0.0, 0.0);
    network.makeFirstInputOfSecond(inhibitory_node, excitatory_node, icon);
    MPILib::DelayedConnection bas(16, 0.36, 0.0);
    network.makeFirstInputOfSecond(baseline_node, excitatory_node, bas);
    network.makeFirstInputOfSecond(baseline_node, inhibitory_node, bas);
}

/**
 * @brief Gating_circuit
 *
 * Takes a source and target node, with a control node and instantiates the
 * rest of nodes necessary to create a gating circuit graph conforming to the
 * Neural Blackboard Architecture.
 */
class Gating_circuit
{
    public:
        // Internal nodes
        NodeId gate_keeper;
        NodeId gate_keeper_e;
        NodeId gate_keeper_b;
        NodeId gate;
        NodeId gate_i;
        NodeId gate_b;
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

    // Configure nodes. Create Gate and Gate Keeper nodes.
    EIB_Population_Pair gate_keeper_eib(network, handler, 1);  // was 250
    gate_keeper = gate_keeper_eib.inhibitory_node;
    gate_keeper_e = gate_keeper_eib.excitatory_node;
    gate_keeper_b = gate_keeper_eib.baseline_node;
    EIB_Population_Pair gate_eib(network, handler, 1);  // was 100
    gate = gate_eib.excitatory_node;
    gate_i = gate_eib.inhibitory_node;
    gate_b = gate_eib.baseline_node;

    // Configure network graph. Instantiate Gating circuit connections.
    MPILib::DelayedConnection con(4, 0.36, 0.0);
    network.makeFirstInputOfSecond(source_assembly, gate, con);
    MPILib::DelayedConnection gcon(4, 0.36, 0.0);
    network.makeFirstInputOfSecond(source_assembly, gate_keeper, gcon);
    MPILib::DelayedConnection icon(60, -0.36, 0.0);
    network.makeFirstInputOfSecond(gate_keeper, gate, icon);
    MPILib::DelayedConnection ccon(60, -0.36, 0.0);
    network.makeFirstInputOfSecond(control, gate_keeper, ccon);
    MPILib::DelayedConnection cong(4, 0.36, 0.0);
    network.makeFirstInputOfSecond(gate, target_assembly, cong);

}

/**
 * @brief Memory_circuit
 *
 * Takes a source and target node and instantiates the
 * rest of nodes necessary to create a memory circuit graph conforming to the
 * Neural Blackboard Architecture. Notice that a memory circuit do not require
 * a control node, since this will be a reverberating population activated by
 * the source and target nodes. The source and target distinction is arbitrary
 * since in this circuit activity flows in both directions. Notice that a
 * memory circuit simply combines two gating circuits and add the connections
 * from the source and target nodes to the control (memory assembly) created.
 */
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

    // Configure network graph
    // Most connections were setup by gating circuits
    // Since the control circuit is an artificial population. We simply add
    // up the rate from source and target nodes to inform the control when to
    // activate.
    MPILib::DelayedConnection con(1, 1, 0.0);
    network.makeFirstInputOfSecond(source_assembly, delay_assembly, con);
    network.makeFirstInputOfSecond(target_assembly, delay_assembly, con);
}


/**
 * @brief Control_circuit
 *
 * Takes a source and target node and two control nodes to instantiate a
 * bidirectional control circuit conforming to the Neural Blackboard Architecture.
 * The source and target distinction is arbitrary since in this circuit activity
 * flows in both directions. Notice that a control circuit is just the
 * combination of two gating circuits.
 */
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

/**
 * @brief BBcell_circuit
 *
 * Takes a source and target node and four control nodes to instantiate a
 * bidirectional compartment circuit conforming to the Neural Blackboard Architecture.
 * The source and target distinction is arbitrary since in this circuit activity
 * can flow in both directions. Notice that a compartment circuit will create
 * necessary nodes to instantiate a gating-memory-gating circuit chain in both
 * directions, connecting the given source and target nodes, constrained
 * by the four control nodes given. To achive this we simply instantiate two
 * control circuits and a memory circuit. For reporting purposes, this class
 * provides a csv description of all nodes with labels and categories.
 */
class BBcell_circuit
{
    public:
        // Internal nodes
        // fc stands for forward circuit and bc for backward circuit
        // the forward and backward distinction just helps distinguish activity
        // from source to target and from target to source, respectively.
        // fc1 and bc1 correspond to linking source main and sub assemblies
        Control_circuit cc1;
        Control_circuit cc2;
        Memory_circuit mc;
        NodeId source_sub_assembly;
        NodeId target_sub_assembly;
        NodeId source_sub_assembly_i;
        NodeId target_sub_assembly_i;
        NodeId source_sub_assembly_b;
        NodeId target_sub_assembly_b;
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
    EIB_Population_Pair source_sub_eib(network, handler, 1);  // was 100
    source_sub_assembly = source_sub_eib.excitatory_node;
    source_sub_assembly_i = source_sub_eib.inhibitory_node;
    source_sub_assembly_b = source_sub_eib.baseline_node;
    EIB_Population_Pair target_sub_eib(network, handler, 1);  // was 100
    target_sub_assembly = target_sub_eib.excitatory_node;
    target_sub_assembly_i = target_sub_eib.inhibitory_node;
    target_sub_assembly_b = target_sub_eib.baseline_node;

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
    node_desc << "source-sub-assembly," << source_sub_assembly << ",sub-assembly" << endl;
    node_desc << "target-sub-assembly," << target_sub_assembly << ",sub-assembly" << endl;
    node_desc << "source-sub-assembly-i," << source_sub_assembly_i << ",circuit-i" << endl;
    node_desc << "target-sub-assembly-i," << target_sub_assembly_i << ",circuit-i" << endl;
    node_desc << "source-sub-assembly-b," << source_sub_assembly_b << ",baseline" << endl;
    node_desc << "target-sub-assembly-b," << target_sub_assembly_b << ",baseline" << endl;

    node_desc << "cc1_forward_gate-keeper," << cc1.forward_circuit.gate_keeper << ",circuit-i" << endl;
    node_desc << "cc1_forward_gate-keeper-e," << cc1.forward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc1_forward_gate-keeper-b," << cc1.forward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc1_forward_gate," << cc1.forward_circuit.gate << ",circuit-e" << endl;
    node_desc << "cc1_forward_gate-i," << cc1.forward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc1_forward_gate-b," << cc1.forward_circuit.gate_b << ",baseline" << endl;

    node_desc << "cc1_backward_gate-keeper," << cc1.backward_circuit.gate_keeper << ",circuit-i" << endl;
    node_desc << "cc1_backward_gate-keeper-e," << cc1.backward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc1_backward_gate-keeper-b," << cc1.backward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc1_backward_gate," << cc1.backward_circuit.gate << ",circuit-e" << endl;
    node_desc << "cc1_backward_gate-i," << cc1.backward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc1_backward_gate-b," << cc1.backward_circuit.gate_b << ",baseline" << endl;

    node_desc << "mc_delay-assembly," << mc.delay_assembly << ",delay" << endl;

    node_desc << "mc_forward_gate-keeper," << mc.forward_circuit.gate_keeper << ",circuit-i" << endl;
    node_desc << "mc_forward_gate-keeper-e," << mc.forward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "mc_forward_gate-keeper-b," << mc.forward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "mc_forward_gate," << mc.forward_circuit.gate << ",circuit-e" << endl;
    node_desc << "mc_forward_gate-i," << mc.forward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "mc_forward_gate-b," << mc.forward_circuit.gate_b << ",baseline" << endl;

    node_desc << "mc_backward_gate-keeper," << mc.backward_circuit.gate_keeper << ",circuit-i" << endl;
    node_desc << "mc_backward_gate-keeper-e," << mc.backward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "mc_backward_gate-keeper-b," << mc.backward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "mc_backward_gate," << mc.backward_circuit.gate << ",circuit-e" << endl;
    node_desc << "mc_backward_gate-i," << mc.backward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "mc_backward_gate-b," << mc.backward_circuit.gate_b << ",baseline" << endl;

    node_desc << "cc2_forward_gate-keeper," << cc2.forward_circuit.gate_keeper << ",circuit-i" << endl;
    node_desc << "cc2_forward_gate-keeper-e," << cc2.forward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc2_forward_gate-keeper-b," << cc2.forward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc2_forward_gate," << cc2.forward_circuit.gate << ",circuit-e" << endl;
    node_desc << "cc2_forward_gate-i," << cc2.forward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc2_forward_gate-b," << cc2.forward_circuit.gate_b << ",baseline" << endl;

    node_desc << "cc2_backward_gate-keeper," << cc2.backward_circuit.gate_keeper << ",circuit-i" << endl;
    node_desc << "cc2_backward_gate-keeper-e," << cc2.backward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc2_backward_gate-keeper-b," << cc2.backward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc2_backward_gate," << cc2.backward_circuit.gate << ",circuit-e" << endl;
    node_desc << "cc2_backward_gate-i," << cc2.backward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc2_backward_gate-b," << cc2.backward_circuit.gate_b << ",baseline" << endl;
}

int main(){
//! [preamble]
    cout << "Demonstrating Blackboard circuit cell" << endl;

// There are three possible states to simulate that should
// be handled by a configuration file. The baseline state in which no input or
// control is necessary. The partial activity state in which only 1 input node
// and control are activated such that the memory assembly remains inactive.
// Finally the complete activity state that reflects the effect of the activation
// of the memory assembly and the whole activation profile of a compartment
// circuit from baseline to activation and back to baseline.

// Nonetheless at the moment there are three copies of this file refliecting
// the 3 circuit simulations hardcoded. activity_circuit.cpp,
// baseline_circuit.cpp and partial_activity_circuit.cpp. The parameters that
// change are the report file name and the list of input and control events.

    // Initialize network and report file for activity simulation
    // Store activity as simple records with time, Nodeid and activity rate.
    // Also store the node description file filled by BBcell_circuit class.
    Network network;
    Report handler("activity", true); // 2 input and 2 control activated
    std::ofstream node_desc;
    node_desc.open("node_desc.csv");

// Configure algorithms

    // Fixed rate algorithm used as dummy for unusued controls
    Rate rate_ext = 0.0;
    RateAlgorithm<MPILib::DelayedConnection> alg_ext(rate_ext);

    // Delay assembly algorithm that will be used by memory circuits
    DelayAssemblyParameter decayparam(1.0, 10.0, 10.0, -10.0);
    DecayAlg decayalg(decayparam);

    // We create the list of input events to configure a Boxcar algorithm
    // These events feed a fixed rate into the connected nodes and adds a
    // desired slope to the events to avoid massive oscilatory behavior due
    // to extreme changes in input rate.
    std::vector<Event> inputevents_source(1);
    inputevents_source[0].start = 0.5;
    inputevents_source[0].end = inputevents_source[0].start + 0.05;
    inputevents_source[0].rate = 10.1;
    BoxcarAlg input_box_source(inputevents_source, 1.0);

    std::vector<Event> inputevents_target(1);
    inputevents_target[0].start = 0.8;
    inputevents_target[0].end = inputevents_target[0].start + 0.05;
    inputevents_target[0].rate = 10.1;
    BoxcarAlg input_box_target(inputevents_target, 1.0);

    // We create the list of control events to configure a Boxcar algorithm
    std::vector<Event> ctrlevents(1);
    ctrlevents[0].start = inputevents_target[0].start + 0.05;
    ctrlevents[0].end = ctrlevents[0].start + 0.15;
    ctrlevents[0].rate = 4.5;
    BoxcarAlg ctrl_box(ctrlevents, 0.01);

    // Population density method algorithm passed to circuit classes to
    // instantiate new nodes.
    // THIS SHOULD BE ELIMINATED FROM THE CLASSES INTERFACE SINCE EIB PAIR
    // IMPLEMENTS ITS ALGORITHM AT THE MOMENT. SO THIS DEFINITION IS NECESSARY
    // HERE FOR COMPATIBILITY BUT MAKES NO DIFFERENCE IN THE CODE.
    // SHOULD ELIMINATE FROM HERE
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
    // SHOULD ELIMINATE UNTIL HERE

// Setup nodes
    // Setup main target and source nodes. For each of them there is an input node
    // feeding events to a reverberating memory node that leads the activity of
    // the main node.
    NodeId input_source = network.addNode(input_box_source, EXCITATORY_DIRECT);
    NodeId input_target = network.addNode(input_box_target, EXCITATORY_DIRECT);

    NodeId source_mem  = network.addNode(decayalg, EXCITATORY_DIRECT);
    NodeId target_mem  = network.addNode(decayalg, EXCITATORY_DIRECT);

    EIB_Population_Pair source_main_eib(network, handler, 1); // was 100
    NodeId source_main_assembly = source_main_eib.excitatory_node;
    NodeId source_main_assembly_i = source_main_eib.inhibitory_node;
    NodeId source_main_assembly_b = source_main_eib.baseline_node;

    EIB_Population_Pair target_main_eib(network, handler, 1); // was 100
    NodeId target_main_assembly = target_main_eib.excitatory_node;
    NodeId target_main_assembly_i = target_main_eib.inhibitory_node;
    NodeId target_main_assembly_b = target_main_eib.baseline_node;

    NodeId ctrl1  = network.addNode(ctrl_box, INHIBITORY_DIRECT);
    NodeId ctrl2  = network.addNode(alg_ext, INHIBITORY_DIRECT);
    NodeId ctrl3  = network.addNode(alg_ext, INHIBITORY_DIRECT);
    NodeId ctrl4  = network.addNode(ctrl_box, INHIBITORY_DIRECT);

    MPILib::DelayedConnection conin(1, 1, 0.0);  // was 20 n and 0.36 eff
    network.makeFirstInputOfSecond(input_source, source_mem, conin);
    network.makeFirstInputOfSecond(input_target, target_mem, conin);
    MPILib::DelayedConnection conmem(4, 0.36, 0.0);  // was 10 n con
    network.makeFirstInputOfSecond(source_mem, source_main_assembly, conmem);
    network.makeFirstInputOfSecond(target_mem, target_main_assembly, conmem);

    // Write node description on file for all created nodes
    node_desc << "node_name," << "node_id," << "node_category" << endl;
    node_desc << "input-source," << input_source << ",trigger" << endl;
    node_desc << "input-target," << input_target << ",trigger" << endl;
    node_desc << "source-mem," << source_mem << ",delay" << endl;
    node_desc << "target-mem," << target_mem << ",delay" << endl;
    node_desc << "source-main-assembly," << source_main_assembly << ",main-assembly" << endl;
    node_desc << "target-main-assembly," << target_main_assembly << ",main-assembly" << endl;
    node_desc << "source-main-assembly-i," << source_main_assembly_i << ",circuit-i" << endl;
    node_desc << "target-main-assembly-i," << target_main_assembly_i << ",circuit-i" << endl;
    node_desc << "source-main-assembly-b," << source_main_assembly_b << ",baseline" << endl;
    node_desc << "target-main-assembly-b," << target_main_assembly_b << ",baseline" << endl;
    node_desc << "control-fc1," << ctrl1 << ",control" << endl;
    node_desc << "control-fc2," << ctrl2 << ",control" << endl;
    node_desc << "control-bc1," << ctrl3 << ",control" << endl;
    node_desc << "control-bc2," << ctrl4 << ",control" << endl;

// Create compartment circuit. Will add the rest of nodes to description file
    BBcell_circuit circuit(network, handler, source_main_assembly,
                           target_main_assembly,
                           ctrl1, ctrl2, ctrl3, ctrl4,
                           alg, decayalg,
                           node_desc);

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
            "activity.log"
        );
    network.configureSimulation(par_run);
    cout << "Circuit setup done. Starting simulation..." << endl;

// Run simulation
    network.evolve();
    node_desc.close();
    return 0;
}