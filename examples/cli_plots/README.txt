Command Line Interface (CLI) Example

******example.xml*******
It is recommended this example is run with an available display (or X-forwarding).
Run the example simulation for a population of Izhikevich neurons

$ python -m miind.run example.xml

Load the MIIND CLI

$ python -m miind.miindio

In the CLI, set the current simulation to example.xml

> sim example.xml

Plot the average firing rate of POP1

> rate POP1

Plot the population density of POP1 at time 0.42 seconds

> plot-density POP1 0.42

Plot the marginal density of POP1 at time 0.42 seconds

> plot-marginals POP1 0.42

If a display was available when running the simulation, generate a density movie of POP1.

> generate-density-movie POP1 512 0.1 pop1_mov

Generate the marginal density movie for POP1

> generate-marginal-movie POP1 512 0.1 pop1_marginal_mov