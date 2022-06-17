A single population of Hodgkin-Huxley neurons.

Default behaviour: Simulate a single population of HH neurons in MIIND and Monte Carlo.

1) Run generateHH.py to build the model and transition matrix files for use with MIIND
* WARNING * This takes a veeeeeery long time (~100 hours) on a single 12 core machine.

2) Run run_hh_finite.py to perform the Monte Carlo simulation of 10000 neurons. 
	Output is the average membrane potential and average gating variables of the population.
3) Run run_hh_miind.py to perform the MIIND simulation.
	Output is the average membrane potential compared to the direct simulation (Monte Carlo).
	
Subsequent runs of run_hh_miind.py will not simulate and only plot the results unless the output directories 
are removed.