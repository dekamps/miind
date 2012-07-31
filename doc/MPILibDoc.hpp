
/** \page MPILibDoc MPILib
 *
 * \section intro_sec Introduction
 * This section gives an introduction to the usage of DynamicLib and presents two examples: a fairly straightforward
 * one, which shows in general how systems of coupled equations can be solved, and a more complex one with
 * a specific application in neuroscience. This example is particularly relevant in the light  of the work done
 * by La Camera et al. (2005).

 * MPINodes represent a significant extension: not only can
 * they query other nodes for their activities, they also contain a reference to an AbstractAlgorithm,
 * which in turn maintains a NodeState. The key idea is that MPINodes can evolve their NodeState
 * by requesting their Algorithms to do so:
 * MPINode has an Evolve() method, which calls its Algorithms Evolve() method. The NodeState describes
 * the state of the node at a certain time \f$ t\f$ and the Algorithm's Evolve() method evolves that node's
 * state over a time \f$\Delta t\f$, which is usually small. MPINodes
 * maintain a list of nodes that connect to them with a list of the weights for every connection.
 * At every time \f$ t\f$, they are able to evaluate the instantaneous contribution of other nodes to
 * itself and that input is passed to the node's Algorithm as a parameter.
 *
 * \image html SimulationLoop.png
 *
 * A MPINode has an AbstractAlgorithm (oval), which operates on a NodeState
 * (rectangle).
 * When prompted by the simulation loop, the AbstractReportHandler sends the current NodeState to a central file. MPINodes are almost
 * autonomous. The central simulation loop determines which Node is in line to evolve its NodeState over a short time step, but the Nodes
 * themselves collect input from other Nodes and deliver this to their own Algorithms which evolve the Node's NodeState. This setup is easy
 * to parallelise see \ref MPI.
 *
 * At the highest level MPINetwork's Evolve() method initiates a loop over all nodes, the simulation loop,
 * in which it requests that every MPINode evolve itself over a short period of time.
 * The MPINetwork does this repeatedly and in such a way a simulation of the network dynamics emerges.
 * A MPINode is also configured with a ReportHandler. At fixed times, the simulation loop queries the
 * MPINode for a Report. The ReportHandler of the MPINode delivers the
 * Report and the Reports are written to disk, so that a record of the simulation is produced.
 * Also, the simulation loop maintains a log file to indicate how far the simulation has progressed and to keep a record
 * of exceptional conditions that occured during simulation.  In the Figure above we show a graphical representation
 * of the classes involved in MPINetwork.
 *
 * Possibly, this sounds somewhat abstract, so we give an example first.
 *
 * \section MPI MPI Parallelisation
 *
 * @todo implement this
 *
 *
 * \section  example_sec Example: Modelling Wilson-Cowan equations
 *
 *
 * Consider a network which consists of two populations, one of which is  described by the Equation below:
 *
 * \f$
 *    \tau \frac{d E}{dt}  =  -E + f(\alpha E  + \epsilon \nu)
 * \f$
 * and one of which simply maintains a fixed output rate and serves as an external population to the network.
 *
 *
 * In the example above we show how a network is set up.
 * First, the WilsonCowanAlgorithms is defined and configured with the appropriate WilsonCowanParameter
 * parameter, which defines the sigmoid parameters.
 * Also, a network needs input: therefore a RateAlgorithm is created, an algorithm whose only action is to
 * set the  NodeState of the MPINode to which it belongs  to a fixed rate (the NodeState consist of a
 * single floating point value in this case).
 * The nodes are then created in the MPINetwork, with their own copy of the WilsonCowanAlgorithm
 * (or RateAlgorithm).
 * A user receives a NodeId as a reference to MPINode that just has been created in the MPINetwork.
 * These NodeIds can then be used
 * to define Connections in the network. After the definition of the MPINetwork, one only has to
 * Configure it and to Evolve it.
 *
 * In this code a standard sigmoid is used: a function of the form:
 * \f$
 * f(x) = \frac{f_{max}}{1 + e^{-\beta x}},
 * \f$
 * where \f$ f_{max} \f$ is the maximum response of the node and \f$\beta\f$ is a noise parameter. In the code above
 * one can see how these parameters are set. There is no need to call or define
 * numerical integrators from the user's point of view.
 * Setting up large networks is a trivial exercise. It just comes down to using AddNode and MakeFirstInputOfSecond
 * repeatedly. A very large network which
 * was to model a neuronal architecture for compositional representations \cite{vdvelde2006} is shown in the Figure below.
 * \image html BBS.png
 */





*/
