A single population of LIF neurons with excitatory and inhibitory conductances.

Default behaviour: Simulate a population of neurons in MIIND and Monte Carlo.

1) Run generateCond3D.py to build the model and transition matrix files at different resolutions for use with MIIND
2) Run run_direct.py to perform the Monte Carlo simulation of 10000 neurons with different input rates. 
	Output is the average membrane potential with a 2Hz input.
3) Run run_miind.py to perform the MIIND simulations.
	First Output is the average membrane potential compared to the direct simulation (Monte Carlo) at 100Hz.
	Second Output is the average firing rate compared to the direct simulation (Monte Carlo) at 100Hz.
4) Run run_current_curves.py to plot the steady membrane potential against input rate for both 
simulation techniques.
	
Subsequent runs of run_miind.py and run_current_curves.py will not simulate and only plot the results unless the output directories 
are removed.