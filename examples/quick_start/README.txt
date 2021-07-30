MIIND Quick Start Example

Files for a quick introduction to MIIND usage. generateCondFiles.py will generate the model and .tmat files for a population of conductance based leaky integrate and fire neurons. cond.xml describes a pair of fully connected E-I populations. First, call generateCondFiles.py, then run cond.xml. 

******generateCondFiles.py*******

$ python generateCondFiles.py

******cond.xml*******
Recommended to run with a display (or X-forwarding).

$ python -m miind.run cond.xml

******Plot the average firing rate of the E population******

$ python -m miind.miindio sim cond.xml
$ python -m miind.miindio rate E