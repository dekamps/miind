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

        Connectivity_Structure(){};

        Connectivity_Structure(double eff, int bcon, int fcon, int icon);
};

Connectivity_Structure::Connectivity_Structure(double eff, int bcon, int fcon,
                                               int icon)
{
    efficacy = eff;
    baseline_connections = bcon;
    forward_connections = fcon;
    inhibitory_connections = icon;
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
    DelayAssemblyParameter decayparam(10.0, baseline, -1.0, -10.0);  // was 500, 80
    DelayAlg delayalg(decayparam);

    // Configure nodes
    baseline_node  = network.addNode(delayalg, EXCITATORY_DIRECT);

    // Configure network graph
    // The graph allows to transform the EI balanced network into a simple
    // forward activity pair of nodes to study the impact of EI balance.
    // We modify here to turn on and off EI mechanisms.
    // Off at the moment (0 efficacy in connections between EI nodes)
    // MPILib::DelayedConnection con(1, 0.0, 0.0);
    // network.makeFirstInputOfSecond(excitatory_node, inhibitory_node, con);
    // MPILib::DelayedConnection icon(1, -0.0, 0.0);
    // network.makeFirstInputOfSecond(inhibitory_node, excitatory_node, icon);

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
                       Algorithm& alg,
                       Connectivity_Structure con_struct);
};

Gating_circuit::Gating_circuit(Network& network, Report& handler,
                               NodeId s_assembly,
                               NodeId t_assembly,
                               NodeId ctrl,
                               Algorithm& alg,
                               Connectivity_Structure con_struct)
{
    cout << "Building gating circuit" << endl;
    source_assembly = s_assembly;
    control = ctrl;
    target_assembly = t_assembly;

    // Configure nodes. Create Gate and Gate Keeper nodes.
    EIB_Population_Pair gate_keeper_eib(network, handler, 1, 0, alg, con_struct);
    gate_keeper = gate_keeper_eib.inhibitory_node;
    // gate_keeper_e = gate_keeper_eib.excitatory_node;
    gate_keeper_b = gate_keeper_eib.baseline_node;
    EIB_Population_Pair gate_eib(network, handler, 1, 1, alg, con_struct);
    gate = gate_eib.excitatory_node;
    // gate_i = gate_eib.inhibitory_node;
    gate_b = gate_eib.baseline_node;

    // Configure network graph. Instantiate Gating circuit connections.
    // EFFICACY AND CONNECTIONS
    MPILib::DelayedConnection con(con_struct.forward_connections,
                                  con_struct.efficacy, 0.0);
    network.makeFirstInputOfSecond(source_assembly, gate, con);
    MPILib::DelayedConnection gcon(con_struct.forward_connections,
                                   con_struct.efficacy, 0.0);
    network.makeFirstInputOfSecond(source_assembly, gate_keeper, gcon);
    MPILib::DelayedConnection icon(con_struct.inhibitory_connections,
                                   -con_struct.efficacy, 0.0);
    network.makeFirstInputOfSecond(gate_keeper, gate, icon);
    MPILib::DelayedConnection ccon(con_struct.inhibitory_connections,
                                   -con_struct.efficacy, 0.0);
    network.makeFirstInputOfSecond(control, gate_keeper, ccon);
    MPILib::DelayedConnection cong(con_struct.forward_connections,
                                   con_struct.efficacy, 0.0);
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
                       Algorithm& alg,
                       DelayAlg delayalg,
                       Connectivity_Structure con_struct);
};

Memory_circuit::Memory_circuit(Network& network, Report& handler,
                               NodeId s_assembly,
                               NodeId t_assembly,
                               Algorithm& alg,
                               DelayAlg delayalg,
                               Connectivity_Structure con_struct)
{
    cout << "Building memory circuit" << endl;
    source_assembly = s_assembly;
    target_assembly = t_assembly;

    // Configure nodes
    delay_assembly  = network.addNode(delayalg, INHIBITORY_DIRECT);

    forward_circuit = Gating_circuit(network, handler, source_assembly,
                                     target_assembly, delay_assembly, alg,
                                     con_struct);
    backward_circuit = Gating_circuit(network, handler, target_assembly,
                                      source_assembly, delay_assembly, alg,
                                      con_struct);

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
                        Algorithm& alg,
                        Connectivity_Structure con_struct);
};

Control_circuit::Control_circuit(Network& network, Report& handler,
                                 NodeId s_assembly,
                                 NodeId t_assembly,
                                 NodeId f_ctrl,
                                 NodeId b_ctrl,
                                 Algorithm& alg,
                                 Connectivity_Structure con_struct)
{
    cout << "Building control circuit" << endl;
    source_assembly = s_assembly;
    target_assembly = t_assembly;
    forward_control = f_ctrl;
    backward_control = b_ctrl;

    // Configure nodes
    forward_circuit = Gating_circuit(network, handler, source_assembly,
                                     target_assembly, forward_control, alg,
                                     con_struct);
    backward_circuit = Gating_circuit(network, handler, target_assembly,
                                      source_assembly, backward_control, alg,
                                      con_struct);

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
                       Algorithm& alg,
                       DelayAlg delayalg,
                       Connectivity_Structure con_struct);
};

BBcell_circuit::BBcell_circuit(Network& network, Report& handler,
                               NodeId s_assembly,
                               NodeId t_assembly,
                               NodeId ctrlfc1,
                               NodeId ctrlfc2,
                               NodeId ctrlbc1,
                               NodeId ctrlbc2,
                               Algorithm& alg,
                               DelayAlg delayalg,
                               Connectivity_Structure con_struct)
{
    cout << "Building blackboard cell circuit" << endl;
    source_main_assembly = s_assembly;
    target_main_assembly = t_assembly;
    control_fc1 = ctrlfc1;
    control_fc2 = ctrlfc2;
    control_bc1 = ctrlbc1;
    control_bc2 = ctrlbc2;

    // Configure nodes
    EIB_Population_Pair source_sub_eib(network, handler, 1, 1, alg, con_struct);
    source_sub_assembly = source_sub_eib.excitatory_node;
    // source_sub_assembly_i = source_sub_eib.inhibitory_node;
    source_sub_assembly_b = source_sub_eib.baseline_node;
    EIB_Population_Pair target_sub_eib(network, handler, 1, 1, alg, con_struct);
    target_sub_assembly = target_sub_eib.excitatory_node;
    // target_sub_assembly_i = target_sub_eib.inhibitory_node;
    target_sub_assembly_b = target_sub_eib.baseline_node;

    // Setup control circuit
    cc1 = Control_circuit(network, handler, source_main_assembly,
                          source_sub_assembly, control_fc1, control_bc1, alg,
                          con_struct);
    // Setup memory circuit
    mc = Memory_circuit(network, handler, source_sub_assembly,
                        target_sub_assembly, alg, delayalg, con_struct);
    // Setup control circuit
    cc2 = Control_circuit(network, handler, target_sub_assembly,
                          target_main_assembly, control_fc2, control_bc2, alg,
                          con_struct);
}

int main(){
// Decide simulation to run
// MODEL: 1 for LIF and 0 for AdEx
// N_EVENTS: 0 for baseline, 1 for partial activity, 2 complete activity
// CONNECTIVITY: 0 for low efficacy, 1 for high efficacy
    int MODEL = 0;
    int N_EVENTS = 0;
    int CONNECTIVITY = 0;
//! [preamble]
    cout << "Demonstrating Blackboard circuit cell" << endl;
    cout << MODEL << " model with " << N_EVENTS << " events with " << CONNECTIVITY << "connectivity" << endl;
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

    string report_name = "activity_mod-" + to_string(MODEL);
    report_name = report_name + "_con-" + to_string(CONNECTIVITY);
    report_name = report_name + "_ev-" + to_string(N_EVENTS);
    Report handler(report_name, true);

    std::ofstream node_desc;
    node_desc.open("node_desc.csv");

// Configure algorithms

    // Fixed rate algorithm used as dummy for unusued controls
    Rate rate_ext = 0.0;
    RateAlgorithm<MPILib::DelayedConnection> alg_ext(rate_ext);

    // Delay assembly algorithm that will be used by memory circuits
    DelayAssemblyParameter decayparam(1.0, 10.0, 10.0, -10.0);
    DelayAlg delayalg(decayparam);

    // We create the list of input events to configure a Boxcar algorithm
    // These events feed a fixed rate into the connected nodes and adds a
    // desired slope to the events to avoid massive oscilatory behavior due
    // to extreme changes in input rate.
    std::vector<Event> inputevents_source(1);
    inputevents_source[0].start = 0.5;
    inputevents_source[0].end = inputevents_source[0].start + 0.05;

    // CHANGE WITH IF
    if(N_EVENTS > 0){
        inputevents_source[0].rate = 10.1;
    }
    else{
        inputevents_source[0].rate = 0.0;
    }

    BoxcarAlg input_box_source(inputevents_source, 1.0);

    std::vector<Event> inputevents_target(1);
    inputevents_target[0].start = 0.8;
    inputevents_target[0].end = inputevents_target[0].start + 0.05;

    // CHANGE WITH IF
    if(N_EVENTS > 1){
        inputevents_target[0].rate = 10.1;
    }
    else{
        inputevents_target[0].rate = 0.0;
    }

    BoxcarAlg input_box_target(inputevents_target, 1.0);

    // We create the list of control events to configure a Boxcar algorithm
    std::vector<Event> ctrlevents(1);
    ctrlevents[0].start = inputevents_target[0].start + 0.05;
    ctrlevents[0].end = ctrlevents[0].start + 0.15;

    // CHANGE WITH IF
    if(N_EVENTS > 0){
        ctrlevents[0].rate = 4.5;
    }
    else{
        ctrlevents[0].rate = 0.0;
    }

    BoxcarAlg ctrl_box(ctrlevents, 0.01);

    // Population density method algorithm passed to circuit classes to
    // instantiate new nodes.
    // THIS SHOULD BE ELIMINATED FROM THE CLASSES INTERFACE SINCE EIB PAIR
    // IMPLEMENTS ITS ALGORITHM AT THE MOMENT. SO THIS DEFINITION IS NECESSARY
    // HERE FOR COMPATIBILITY BUT MAKES NO DIFFERENCE IN THE CODE.
    // SHOULD ELIMINATE FROM HERE
  std::vector<std::string> vec_mat_0{"aexp045c42a8-7be8-409a-a37e-a88e1d135c03_1_0_0_0_.mat",
                  "aexp045c42a8-7be8-409a-a37e-a88e1d135c03_-1_0_0_0_.mat",
                  "aexp045c42a8-7be8-409a-a37e-a88e1d135c03_3_0_0_0_.mat",
                  "aexp045c42a8-7be8-409a-a37e-a88e1d135c03_-3_0_0_0_.mat"};
  TwoDLib::MeshAlgorithm<DelayedConnection> alg("aexp045c42a8-7be8-409a-a37e-a88e1d135c03.model", vec_mat_0, 0.00002);

//Setup connectivity structure
    double eff;
    int bcon;
    int fcon;
    int icon;

    if(CONNECTIVITY == 0 && MODEL == 1){
        eff = 0.1;
        bcon = 80;
        fcon = 30;
        icon = 90;
    }
    else if (CONNECTIVITY == 1 && MODEL == 1){
        eff = 0.3;
        bcon = 25;
        fcon = 5;
        icon = 80;
    }
    else if (CONNECTIVITY == 0 && MODEL == 0){
        eff = 1;
        bcon = 80;
        fcon = 30;
        icon = 90;
    }
    else if (CONNECTIVITY == 1 && MODEL == 0){
        eff = 3;
        bcon = 25;
        fcon = 5;
        icon = 80;
    }
    Connectivity_Structure con_struct(eff, bcon, fcon, icon);

// Setup nodes
    // Setup main target and source nodes. For each of them there is an input node
    // feeding events to a reverberating memory node that leads the activity of
    // the main node.
    NodeId input_source = network.addNode(input_box_source, EXCITATORY_DIRECT);
    NodeId input_target = network.addNode(input_box_target, EXCITATORY_DIRECT);

    NodeId source_mem  = network.addNode(delayalg, EXCITATORY_DIRECT);
    NodeId target_mem  = network.addNode(delayalg, EXCITATORY_DIRECT);

    EIB_Population_Pair source_main_eib(network, handler, 1, 1, alg, con_struct);
    NodeId source_main_assembly = source_main_eib.excitatory_node;
    // NodeId source_main_assembly_i = source_main_eib.inhibitory_node;
    NodeId source_main_assembly_b = source_main_eib.baseline_node;

    EIB_Population_Pair target_main_eib(network, handler, 1, 1, alg, con_struct);
    NodeId target_main_assembly = target_main_eib.excitatory_node;
    // NodeId target_main_assembly_i = target_main_eib.inhibitory_node;
    NodeId target_main_assembly_b = target_main_eib.baseline_node;

    NodeId ctrl1  = network.addNode(ctrl_box, INHIBITORY_DIRECT);
    NodeId ctrl2  = network.addNode(alg_ext, INHIBITORY_DIRECT);
    NodeId ctrl3  = network.addNode(alg_ext, INHIBITORY_DIRECT);
    NodeId ctrl4  = network.addNode(ctrl_box, INHIBITORY_DIRECT);

    MPILib::DelayedConnection conin(1, 1, 0.0);
    network.makeFirstInputOfSecond(input_source, source_mem, conin);
    network.makeFirstInputOfSecond(input_target, target_mem, conin);

    // EFFICACY AND CONNECTIONS
    MPILib::DelayedConnection conmem(con_struct.forward_connections,
                                     con_struct.efficacy, 0.0);

    network.makeFirstInputOfSecond(source_mem, source_main_assembly, conmem);
    network.makeFirstInputOfSecond(target_mem, target_main_assembly, conmem);

// Create compartment circuit. Will add the rest of nodes to description file
    BBcell_circuit bb(network, handler, source_main_assembly,
                      target_main_assembly,
                      ctrl1, ctrl2, ctrl3, ctrl4,
                      alg, delayalg, con_struct);

    // Write node description on file for all created nodes
    node_desc << "node_name," << "node_id," << "node_category" << endl;
    node_desc << "input-source," << input_source << ",trigger" << endl;
    node_desc << "input-target," << input_target << ",trigger" << endl;
    node_desc << "source-mem," << source_mem << ",delay" << endl;
    node_desc << "target-mem," << target_mem << ",delay" << endl;
    node_desc << "source-main-assembly," << source_main_assembly << ",main-assembly" << endl;
    node_desc << "target-main-assembly," << target_main_assembly << ",main-assembly" << endl;
    // node_desc << "source-main-assembly-i," << source_main_assembly_i << ",circuit-i" << endl;
    // node_desc << "target-main-assembly-i," << target_main_assembly_i << ",circuit-i" << endl;
    node_desc << "source-main-assembly-b," << source_main_assembly_b << ",baseline" << endl;
    node_desc << "target-main-assembly-b," << target_main_assembly_b << ",baseline" << endl;
    node_desc << "control-fc1," << ctrl1 << ",control" << endl;
    node_desc << "control-fc2," << ctrl2 << ",control" << endl;
    node_desc << "control-bc1," << ctrl3 << ",control" << endl;
    node_desc << "control-bc2," << ctrl4 << ",control" << endl;

    // Write node description on file
    node_desc << "source-sub-assembly," << bb.source_sub_assembly << ",sub-assembly" << endl;
    node_desc << "target-sub-assembly," << bb.target_sub_assembly << ",sub-assembly" << endl;
    // node_desc << "source-sub-assembly-i," << bb.source_sub_assembly_i << ",circuit-i" << endl;
    // node_desc << "target-sub-assembly-i," << bb.target_sub_assembly_i << ",circuit-i" << endl;
    node_desc << "source-sub-assembly-b," << bb.source_sub_assembly_b << ",baseline" << endl;
    node_desc << "target-sub-assembly-b," << bb.target_sub_assembly_b << ",baseline" << endl;

    node_desc << "cc1_forward_gate-keeper," << bb.cc1.forward_circuit.gate_keeper << ",circuit-i" << endl;
    // node_desc << "cc1_forward_gate-keeper-e," << bb.cc1.forward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc1_forward_gate-keeper-b," << bb.cc1.forward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc1_forward_gate," << bb.cc1.forward_circuit.gate << ",circuit-e" << endl;
    // node_desc << "cc1_forward_gate-i," << bb.cc1.forward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc1_forward_gate-b," << bb.cc1.forward_circuit.gate_b << ",baseline" << endl;

    node_desc << "cc1_backward_gate-keeper," << bb.cc1.backward_circuit.gate_keeper << ",circuit-i" << endl;
    // node_desc << "cc1_backward_gate-keeper-e," << bb.cc1.backward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc1_backward_gate-keeper-b," << bb.cc1.backward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc1_backward_gate," << bb.cc1.backward_circuit.gate << ",circuit-e" << endl;
    // node_desc << "cc1_backward_gate-i," << bb.cc1.backward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc1_backward_gate-b," << bb.cc1.backward_circuit.gate_b << ",baseline" << endl;

    node_desc << "mc_delay-assembly," << bb.mc.delay_assembly << ",delay" << endl;

    node_desc << "mc_forward_gate-keeper," << bb.mc.forward_circuit.gate_keeper << ",circuit-i" << endl;
    // node_desc << "mc_forward_gate-keeper-e," << bb.mc.forward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "mc_forward_gate-keeper-b," << bb.mc.forward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "mc_forward_gate," << bb.mc.forward_circuit.gate << ",circuit-e" << endl;
    // node_desc << "mc_forward_gate-i," << bb.mc.forward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "mc_forward_gate-b," << bb.mc.forward_circuit.gate_b << ",baseline" << endl;

    node_desc << "mc_backward_gate-keeper," << bb.mc.backward_circuit.gate_keeper << ",circuit-i" << endl;
    // node_desc << "mc_backward_gate-keeper-e," << bb.mc.backward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "mc_backward_gate-keeper-b," << bb.mc.backward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "mc_backward_gate," << bb.mc.backward_circuit.gate << ",circuit-e" << endl;
    // node_desc << "mc_backward_gate-i," << bb.mc.backward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "mc_backward_gate-b," << bb.mc.backward_circuit.gate_b << ",baseline" << endl;

    node_desc << "cc2_forward_gate-keeper," << bb.cc2.forward_circuit.gate_keeper << ",circuit-i" << endl;
    // node_desc << "cc2_forward_gate-keeper-e," << bb.cc2.forward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc2_forward_gate-keeper-b," << bb.cc2.forward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc2_forward_gate," << bb.cc2.forward_circuit.gate << ",circuit-e" << endl;
    // node_desc << "cc2_forward_gate-i," << bb.cc2.forward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc2_forward_gate-b," << bb.cc2.forward_circuit.gate_b << ",baseline" << endl;

    node_desc << "cc2_backward_gate-keeper," << bb.cc2.backward_circuit.gate_keeper << ",circuit-i" << endl;
    // node_desc << "cc2_backward_gate-keeper-e," << bb.cc2.backward_circuit.gate_keeper_e << ",circuit-e" << endl;
    node_desc << "cc2_backward_gate-keeper-b," << bb.cc2.backward_circuit.gate_keeper_b << ",baseline" << endl;
    node_desc << "cc2_backward_gate," << bb.cc2.backward_circuit.gate << ",circuit-e" << endl;
    // node_desc << "cc2_backward_gate-i," << bb.cc2.backward_circuit.gate_i << ",circuit-i" << endl;
    node_desc << "cc2_backward_gate-b," << bb.cc2.backward_circuit.gate_b << ",baseline" << endl;

// Setup simulation
    const SimulationRunParameter
        par_run
        (
            handler,
            10000000,
            0.0,  // start time
            0.05,  // end time
            1e-3,  // report step
            1e-4,  // simulation step
            "activity.log",
            1e-03 // ???
        );
    network.configureSimulation(par_run);
    cout << "Circuit setup done. Starting simulation..." << endl;

// Run simulation
    network.evolve();
    node_desc.close();
    return 0;
}