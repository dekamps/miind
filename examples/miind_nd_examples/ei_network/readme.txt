E-I Network of LIF neurons with excitatory and inhibitory conductances.

Default behaviour: Simulate an E-I network of neurons with a 50/50 ratio and 80/20 ratio of excitatory/inhibitory 
connections in MIIND and Monte Carlo.

1) Run generateCond3D.py to build the model and transition matrix files for use with MIIND
2) Run run_direct_EI.py to perform the Monte Carlo simulation of 10000 neurons. 
	Output is the average membrane potential of E.
3) Run run_miind_EI.py to perform the MIIND simulations.
	First Output is the average membrane potential of E compared to the direct simulation (Monte Carlo).
	Second Output is the average firing rate of E compared to the direct simulation (Monte Carlo).
	
Subsequent runs of run_miind_EI.py will not simulate and only plot the results unless the output directories 
are removed.