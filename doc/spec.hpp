/**
\page page_specifications Design Specifications for MIIND's workflow

\section sec_workflow_general General Considerations
\subsection sec_network_creation Network Creation

MIIND is designed to simulate circuits of neural populations, and provides tools to analyse the simulation results. Simulations
can be described in C++ and XML and a Python interface is foreseen. Here, we describe the workflow in a platform independent way.
Below, we will describe the design specifications for each platform.

The concept of a network is central to MIIND. From the user perspective a network is a directed graph, where we associated
each vertex with a \e Node and each edge with a \e Connection. 
The type of \e Connection must be stated for a simulation, and its definition is the first step in any workflow.
A <em>Connection Type</em> can be a single floating point number, or something more complex, like a tuple of floating points.
Each \e Node is associated with an \e Algorithm; each edge is associated with a \e Connection. 

After the specification of the connection type, the network is built. This happens in several steps, described here.
In the MIIND workflow \e Algorithm instances are selected  first. Each \e Algorithm instance has a type; its type determines its state and the algorithm
used to evolve that state. An \e Algorithm has an \e AlgorithmParameter. Two \e Algorithm instances can be of the same type, 
but can differ in their <em>AlgorithmParameters</em>. For example, a WilsonCowanAlgorithm is of a different type than a GeomAlgorithm; two GeomAlgorithm 
instances may or may not have different GeomParameters. \e AlgorithmParameters can be named, and if they are named they can be shared by different <em>Algorithm Instances</em>.
If they are not named, they are an implicit part of the \e Algorithm definition, and cannot be shared.

Each \e Algorithm instance
has a \e Name (an algorithm name), that is unique within the simulation. A \e Node has a unique identifier: an non-negative integer number, reflecting the order
in which they were created. A \e Node can have 
a \e Name (a node name) that must be unique within the simulation: no other node may have that name. A \e Node is associated with one and only one <em>Algorithm Instance</em>;
different \e Nodes may be associated with the same <em>Algorithm Instance</em>. An \e Algorithm can have \e system parameter, which selects a particular implementation
of an \e Algorithm, when mutiple implementations are available. It can also have a \e version, when multiple versions of an implementation are provided.

 \e Connections are used to pass \e NodeStates,
a single floating point number, from one \e Node to another.  a \e Node can be connected to many other \e Nodes via \e Connections. A \e Node can ask its \e Algorithm
to interpret its input and all \e Algorithms in the network must be \e compatible with the type of \e Connection. The simulation user is responsible for selecting the  
\e Connection type and for selection \e Algorithm types that are consistent with the \e Connection type.

\e Node instances  are created in order and are numbered uniquely in order of creation. Upon creation, a \e Node must be associated with an \e Algorithm instance. After
all nodes have been created, nodes are connected: for a \e Connection, the input and output node are specified and also a tuple of floating points representing its value.
Upon completion of the connections, the network is considered to be complete. 

Some workflows will require an explicit instantiation of a \e Network instance -e.g. in C++ it must be instantiated, 
but it is not required by this specification as all \e Nodes within one simulation
are required to refer to the same network, and the collection of all nodes and connections determine the network.

\subsection sec_simulation_running Simulation Running
A \e Simulation is specified by a \e SimulationParameter, which species
- a floating point \e begin_time.
- a floating point \e end_time.
- a string \e base_name: this is the name that will be used for the log file, which will be '\<basename\>.log' and the simulation results. 
- a floating point  \e step_time, specifying the basic time step of the simulation; this parameter requires special consideration and may phased out (see the discussion in \ref sec_step_time).
- a boolean onscreen determines whether online visualization should be used (default: false). 
This is a hint, in some workflows online visualization may not make sense; consult concrete workflows.
- a state object, describing whether which densities should be written out when (see below)
- a floating point \e report_time, when the simulation time exceeds a multiple of report time firing rates, and where appropriate densities are written into the simulation file.

The state object contains the following variables:
- a boolean withstate determines whether densities should be written out or not (default: false)
- a list of node names, or node ids, that specifies the densities of which nodeswill be written out (default: all)
- a time interval during which densities will be written out (default all)

\subsection sec_step_time The network time step
It is clear that the network simulation as a whole must proceed in a fixed time step. Geometric binning algorithms are created with a certain time step as parameter.
For now, we will require that all \e Algorithms are able to report this parameter, and that the network checks whether all \e Algorithms use the same time step.
If this procedure is acceptable, there is no need for a network time step parameter as user will be reminded to use consistent meshes. In future use, the following consideration
is possible: meshes take considerable effort to generate, and if the time steps are mutiple of each other, a slighly more complex simulation loop might be in order.
In that case a network time step parameter makes sense, as it allows the network to make checks about the sensibility of different time steps provided by each simulation. Hence,
it will not be phased out for now. 

So the following erquirements will be enforced: Algorithms will report their time step;the network will enforce that all algorithms use the same time step and that this is 
the network time step; the workflow will flag a user attempt to set a different network step time as an error, and will not perform simulation. 

\subsection sec_sim_results Simulation Results
The simulation log and a firing rates results file will be produced in the same directory where the executable is run. The firing rates are stored in one results file.
The file can be organised per node, with a graph for each node, where the graph is labeled by the node name provided by the user; if the user didn't provide on by the node's
creation id. Alternatively, the file can be organised in time slices with a firing rate given for each node, for each time slice, again labeled by node name if the user
provided one.

Density results for one dimensional populations can be written into the same file as the firing rates. Densities must be labeled by node and time slice.
Due to the large amount of information present in two dimensional densities, it is necessary to write each density in its own file. We require that densities
be written to a separate directory and that within that directory density files are labeled by node (name or id when name not provided and time slice.

\section sec_design_xml Design of the XML Workflow
The XML workflow is a concrete instantiation of the design descibed in the last section.

- A simulation is delimited by a "<Simulation> ... </Simulation>" tags.
- The type of a \e Connection is specified by a WeightType definition, e.g. "<WeightType>DelayedConnection</WeightType>". 
- The XML specification of \e Algorithms and \e AlgorithmParameters matches the C++ class definitions one-on-one, i.e. the XML syntax for 
  the following definition of a NeuronParameter:
  \verbatim
  <NeuronParameter>
  <t_membrane>50e-3</t_membrane>
  <t_refractive>0</t_refractive>
  <V_threshold>1.0</V_threshold>
  <V_reset>0</V_reset>
  <V_reversal>0</V_reversal>
  </NeuronParameter>  
  \endverbatim
  can directly be inferred from the C++ interface from GeomLib::NeuronParameter. So any algorithm can be constructed from \ref page_available_algorithms which links through to
the relevant C++ definitions.
*/

