Leaky Integrate and Fire Python Sweep Interface

******sweep_one_shot.py*******
Run several lif population simulations with differing efficacies (and input rates to keep the average input the same). As the efficacy increases, there is greater noise and the average firing rate is pushed upward. 

sweep_one_shot runs all simulations and plots the results.

$ python sweep_one_shot.py

******sweep_re_runnable.py*******
Run several lif population simulations with differing efficacies (and input rates to keep the average input the same). As the efficacy increases, there is greater noise and the average firing rate is pushed upward. 

sweep_re_runnable runs all simulations and records the results in separate output directories. When sweep_re_runnable.py is run again, the results are plotted without re-running the simulations.

$ python sweep_re_runnable.py