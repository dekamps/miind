/**
 * \page sub_page_xml_example An XML Example
 *
 * \section Introduction
 * Make sure you have read the \ref workflow section. It explains how a Python script van be used to turn MIIND XML files into executables.
 * The XML file discussed here, relates to \ref page_examples. This particular XML file resides in the python directory, immediately
 * below the top-level directory and is called omurtag.xml. It can be ran immediately as is, as described in \ref workflow.
 *
 * We will replicate the experiment discussed in section \ref page_examples. The original paper considered a large population of leaky-integrate-and-fire    
 * (LIF) neurons. Membrane parameters are rescaled and dimensionless: reset and reversal potential are 0, threshold is one.                                  
 * Each neuron in this population receives a Poisson distributed spike train of 800 Hz. The synapses are delta-synapses with                                 
 * a synaptic efficacy of $h=0.03$, i.e. each inciming spike raises the membrane potential of the postsynaptic neuron by 3 percent                           
 * of the threshold potential. All neurons are at rest initially. Input is switched on at $t=0$. <i>Each neuron sees spike trains that                       
 * are individually different</i>. Such a simulation can easily be set up in NEST or BRIAN and the results can be inspected in the spike                     
 * raster of section \ref page_examples. The set up in MIIND is not very different. 
 * 
 * \section sec_xml_file  The XMLfile.
 * Inspect the <a href="http://miind.sf.net/ourtag.xml">XML file</a>. Most browsers will allow you to expand or collapse the nodes. At the highest 
 * level the file looks like this:
 * \image html xml_top.png
 *
 * \subsection subsec_xml_weighttype The WeightType
 * The <WeightType> tag specifies the type of connections in the network. At the moment two types are supported: double and DelayedConnection.
 * double is simple a numerical value, DelayedConnection is a tuple of three numerical values: number of connections; efficacy; delay.
 * The algorithms specified later in the file must be compatible with the connection type. As a rule of the thumb, always used DelayedConnection,
 * except for very simple WilsonCowan-type simulations. In general start working from an existing XML template. The C++ documentation is authorative
 * in case of doubt.
 *
 * \subsection subsec_xml_algorithm  The Algorithms
 * Expanding the algorithms, we see that they are all a kind of AbstractAlgorithm of a certain type, and that there is a corresponding tag
 * for the type, which among other things contain a name. The name will later be used by nodes to refer to a particular algorithm. The
 * RateAlgorithm is an algorithm that functions as an external input to the network, providing a constant background rate. The GeomAlgorithm
 * entries are population density algorithms, configured to represent leaky-integrate-and-fire neurons. The GeomAlgorithm will be discussed in detail
 * elsewhere. Instead of GeomAlgorithm we could have used WilsonCowan algorithm for a much simplified simulation. Consult the WilsonCowan example.
 * You will see that apart from the configuration of te algorithms, it look nearly identical to this example.
 * \image html xml_alg.png
 *
 * \subsection subsec_xml_nodes The Nodes
 * Collapsing the algorithms and expanding the nodes, the file looks as follows:
 * \image html xml_nodes.png
 * Each node is configured with an algorithm. This is achieved by identifying the algorithm field of the node with the name of an algorithm.
 * Furthermore, a node is made EXCITATORY, INHIBITORY or NEUTRAL. This allows a check on Dale's law which states that a neuron that
 * is excitatory (inhibitory) on one node is excitatory (inhibitory) on all. If a node is declared neutral, no such check is made.
 * If a node is declared to  be GAUSSIAN, its contribution to other nodes will be interpreted as a Gaussian white noise input. Each
 * node has its own name, that will be used to define connections.
 * \subsection subsec_xml_io The SimulationIO section
 * Non parallel simulations, i.e. those not using MPI can be monitored witha running histogram of the state and rate variables of selected nodes.
 * This feature is useful in circuit design and debugging. For the moment this deactivated in the release version, but it will be back in soon.
 * At present the most important thing is the name of the simulation results file is that specified here.
 * \image html xml_io.png
 * \subsection subsec_xml_simpar Configuring the Simulation
 * When the SimulationRunParameter node is expanded, the following items are visible:
 * \image html xml_simpar.png
 * The interpretation is straightforward:
 * - max_iter: the maximum number of iterations the network is allowed to perform. Simply a large number. Likely to be phased out.
 * - t_begin: start time of the simulation
 * - t_end: stop time of the simulation
 * - t_step: the time step of the simulation. Algorithms are guaranteed to advance their node state by at least their value. Internally they may use different parameters, if this is specified in the configuration of the algorithm
 * - t_report firing rates are report after the simulation has progressed by a unit of t_report
 * - t_update Update time of the life canvas.
 * - t_state_report If the full state of the density is written out a lot of data will be generated,a nd one may opt to do this at a coarser
 * time step than for the firing rates
 * - name_log name of the log file
 */
