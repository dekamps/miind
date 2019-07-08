# MIIND website
## The Barcelona tutorial (13/07/2019)
On Monday 8 July 2017 documentation for the MIIND tutorial at CNS2019 in Barcelona will appear here. It is strongly recommended you bring a laptop, we will bring virtual machines
with MIIND preinstalled, so if you can run Virtual Box or similar, you will be able to participate. If you run Ubuntu, a Debbian package will be made available,
that will run faster.


<img src="https://github.com/dekamps/miind/blob/master/images/AdExp.gif" alt="drawing" width="400"/>


## MIIND: a population level simulator.
What is MIIND? It is a simulator that allows the creation, simulation and analysis of large-scale neural networks. It does not model individual neurons, but models populations
directly, similarly to a neural mass models, except that we use population density techniques. Population density techniques are based on point model neurons, such as
leaky-integrate-and-fire (LIF), quadratic-integrate-and-fire neurons (QIF), or more complex ones, such as adaptive-exponential-integrate-and-fire (AdExp), Izhikevich,
Fitzhugh-Nagumo (FN). MIIND is able to model populations of 1D neural models (like LIF, QIF), or 2D models (AdExp, Izhikevich, FN, others). It does so by using
statistical techniques to answer the question: "If I'd run a NEST or BRIAN simulation (to name some point model-based simulators), where in state space would my neurons be?"
We calculate this distribution in terms of a density function, and from this density function we can infer many properties of the population, including its own firing rate.
By modeling large-scale networks as homogeneous populations that exchange firing rate statistics, rather than spikes, remarkable efficiency can be achieved, whilst retaining
a connection to spiking neurons that is not present in neural mass models.

## Getting started
### Obtaining and Installing MIIND

###
The <a href="https://docs.google.com/document/d/1e9OK_9YiG7MusgeuAgGj_JiIOZPyh1mj7Mqh-D-Nszo/edit#">CNS Tutorial Materials</a>. 

## Gallery
### Single Population: Fitzhugh-Nagumo (Grid method)
[![](http://img.youtube.com/vi/vv9lgntZhYQ/0.jpg)](http://www.youtube.com/watch?v=vv9lgntZhYQ "MIIND : Fitzhugh-Nagumo example")

### Izhikevich
[![](http://img.youtube.com/vi/8p7jEz-qWTY/0.jpg)](http://www.youtube.com/watch?v=8p7jEz-qWTY "MIIND : Izhikevich example")

### Replication of Half Center Central Pattern Generator
[![](http://img.youtube.com/vi/9pC4MOWQ-Ho/0.jpg)](http://www.youtube.com/watch?v=9pC4MOWQ-Ho "MIIND : Persistent Sodium Half Centre example")
