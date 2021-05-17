MIIND LIF GeomAlgorithm Example

A legacy 1D version of the MIIND population density technique for a population of LIF neurons (GeomAlgorithm). 

Requires ROOT, MIIND built and installed from source, Linux MIIND Standalone

******response.xml*******
Run the simulation.

$ miindio.py

> sim response.xml
> submit
> run

******response_analyse.py*******
Generate a .dat file containing the analytic average output for the population.

$ python response_analyse.py

******analyse_response.py*******
Generate a .dat file contining the response average output rate from MIIND.

$ python analyse_rate.py

******response_curve.py*******
Plot the response curve for multiple inputs

$ python response_curve.py