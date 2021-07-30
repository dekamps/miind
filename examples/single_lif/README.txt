MIIND LIF GeomAlgorithm Example 2

A legacy 1D version of the MIIND population density technique for a population of LIF neurons (GeomAlgorithm). 

Requires ROOT, MIIND built and installed from source, Linux MIIND Standalone

******single_lif.py*******
Generate three separate simulation XML files : no_decay.xml, no_input.xml, response.xml

$ python single_lif.py

******no_decay.xml, no_input.xml, response.xml*******
Run the simulations.

$ miindio.py

> sim no_decay.xml
> submit
> run

> sim no_input.xml
> submit
> run

> sim response.xml
> submit
> run

******analyse_single_lif.py*******
Plot the output activity from all three simulations.

$ python analyse_single_lif.py
