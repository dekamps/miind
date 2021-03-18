# MIIND: a population level simulator.

MIIND is a simulator that allows the creation, simulation and analysis of large-scale neural networks. It does not model individual neurons, but models populations
directly, similarly to a neural mass models, except that we use population density techniques. Population density techniques are based on point model neurons, such as
leaky-integrate-and-fire (LIF), quadratic-integrate-and-fire neurons (QIF), or more complex ones, such as adaptive-exponential-integrate-and-fire (AdExp), Izhikevich,
Fitzhugh-Nagumo (FN). MIIND is able to model populations of 1D neural models (like LIF, QIF), or 2D models (AdExp, Izhikevich, FN, others). It does so by using
statistical techniques to answer the question: "If I'd run a NEST or BRIAN simulation (to name some point model-based simulators), where in state space would my neurons be?"
We calculate this distribution in terms of a density function, and from this density function we can infer many properties of the population, including its own firing rate.
By modeling large-scale networks as homogeneous populations that exchange firing rate statistics, rather than spikes, remarkable efficiency can be achieved, whilst retaining
a connection to spiking neurons that is not present in neural mass models.

## Version 1.0.8 (03/2021)

MIIND is now available through python pip!

## Three dimensional population density methods! (26/11/2019)
They said it could not be done, but we have created an efficient version of the Hindmarsh rose model,
a neural model with three state variables.
<img src="https://github.com/dekamps/miind/blob/master/images/hindmarsh.gif" alt="drawing" width="400"/>

## Gallery
### Single Population: Fitzhugh-Nagumo (Mesh Method)
[![](http://img.youtube.com/vi/vv9lgntZhYQ/0.jpg)](http://www.youtube.com/watch?v=vv9lgntZhYQ "MIIND : Fitzhugh-Nagumo example")

### Izhikevich
[![](http://img.youtube.com/vi/8p7jEz-qWTY/0.jpg)](http://www.youtube.com/watch?v=8p7jEz-qWTY "MIIND : Izhikevich example")

### Adaptive Exponential Integrate and Fire
<img src="https://github.com/dekamps/miind/blob/master/images/AdExp.gif" alt="drawing" width="400"/>

### Replication of Half Center Central Pattern Generator
[![](http://img.youtube.com/vi/9pC4MOWQ-Ho/0.jpg)](http://www.youtube.com/watch?v=9pC4MOWQ-Ho "MIIND : Persistent Sodium Half Centre example")
