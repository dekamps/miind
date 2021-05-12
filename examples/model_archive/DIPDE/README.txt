LIF Population Models for comparison with DIPDE

******single.xml*******
Simulate a population of LIF neurons with an excitatory input

$ python -m miind.run single.xml

******recurrent.xml*******
Simulate a population of LIF neurons with an excitatory input and recurrent connections

$ python -m miind.run single.xml

******ei.xml*******
Simulate a population of LIF neurons with both an excitatory and inhibitory input

$ python -m miind.run ei.xml

******variable.xml*******
Simulate a population of LIF neurons with a variable excitatory input
Requires MIIND built and installed from source, Linux

$ miindio.py sim variable.xml
$ miindio.py submit
$ miindio.py run

