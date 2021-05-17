.. MIIND documentation master file, created by
   sphinx-quickstart on Wed Mar 17 07:54:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MIIND: a population level simulator
==============================================

.. raw:: html

    <div style="position: relative; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/vv9lgntZhYQ?autoplay=1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

**MIIND** is a simulator that allows the creation, simulation and analysis of large-scale neural networks. It does not model individual neurons, but models populations
directly, similarly to a neural mass models, except that we use population density techniques. Population density techniques are based on point model neurons, such as
leaky-integrate-and-fire (LIF), quadratic-integrate-and-fire neurons (QIF), or more complex ones, such as adaptive-exponential-integrate-and-fire (AdExp), Izhikevich,
Fitzhugh-Nagumo (FN). MIIND is able to model populations of 1D neural models (like LIF, QIF), or 2D models (AdExp, Izhikevich, FN, others). It does so by using
statistical techniques to answer the question: "If I'd run a NEST or BRIAN simulation (to name some point model-based simulators), where in state space would my neurons be?"
We calculate this distribution in terms of a density function, and from this density function we can infer many properties of the population, including its own firing rate.
By modeling large-scale networks as homogeneous populations that exchange firing rate statistics, rather than spikes, remarkable efficiency can be achieved, whilst retaining
a connection to spiking neurons that is not present in neural mass models.

.. toctree::
   :maxdepth: 2
   :caption: MIIND
   
   installation
   quickstart
   publications



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
