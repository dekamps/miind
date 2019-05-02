# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
Description of models

"""

from .base import ModelNumbaDfun, LOG, numpy, basic, arrays
from numba import guvectorize, float64

from mpi4py import MPI

import sys
sys.path.insert(0, '/home/hugh/dev/miind/build/libs/PythonModels/wilsoncowan/')

import libmiindwc


class Miind_WilsonCowan(ModelNumbaDfun):
    r"""
    **References**:

    .. [WC_1972] Wilson, H.R. and Cowan, J.D. *Excitatory and inhibitory
        interactions in localized populations of model neurons*, Biophysical
        journal, 12: 1-24, 1972.

    """

    _ui_name = "Wilson-Cowan (MIIND)"
    ui_configurable_parameters = ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'tau_e', 'tau_i',
                                  'r_e',  'r_i',  'k_e',  'k_i',  'P',  'Q',
                                  'b_e', 'b_i', 'a_e', 'a_i']

    # Define traited attributes for this model, these represent possible kwargs.
    c_ee = arrays.FloatArray(
        label=":math:`c_{ee}`",
        default=numpy.array([12.0]),
        range=basic.Range(lo=11.0, hi=16.0, step=0.01),
        doc="""Excitatory to excitatory  coupling coefficient""",
        order=1)

    c_ie = arrays.FloatArray(
        label=":math:`c_{ei}`",
        default=numpy.array([4.0]),
        range=basic.Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to excitatory coupling coefficient""",
        order=2)

    c_ei = arrays.FloatArray(
        label=":math:`c_{ie}`",
        default=numpy.array([13.0]),
        range=basic.Range(lo=2.0, hi=22.0, step=0.01),
        doc="""Excitatory to inhibitory coupling coefficient.""",
        order=3)

    c_ii = arrays.FloatArray(
        label=":math:`c_{ii}`",
        default=numpy.array([11.0]),
        range=basic.Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to inhibitory coupling coefficient.""",
        order=4)

    tau_e = arrays.FloatArray(
        label=r":math:`\tau_e`",
        default=numpy.array([10.0]),
        range=basic.Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Excitatory population, membrane time-constant [ms]""",
        order=5)

    tau_i = arrays.FloatArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10.0]),
        range=basic.Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Inhibitory population, membrane time-constant [ms]""",
        order=6)

    b_e = arrays.FloatArray(
        label=":math:`b_e`",
        default=numpy.array([2.8]),
        range=basic.Range(lo=1.4, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of the excitatory sigmoid function""",
        order=7)

    b_i = arrays.FloatArray(
        label=r":math:`b_i`",
        default=numpy.array([4.0]),
        range=basic.Range(lo=2.0, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of a sigmoid function [in
        threshold units]""",
        order=8)

    r_e = arrays.FloatArray(
        label=":math:`r_e`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Excitatory refractory period""",
        order=9)

    r_i = arrays.FloatArray(
        label=":math:`r_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Inhibitory refractory period""",
        order=10)

    k_e = arrays.FloatArray(
        label=":math:`k_e`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Maximum value of the excitatory response function""",
        order=11)

    k_i = arrays.FloatArray(
        label=":math:`k_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Maximum value of the inhibitory response function""",
        order=12)

    P = arrays.FloatArray(
        label=":math:`P`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the excitatory population.
        Constant intensity.Entry point for coupling.""",
        order=13)

    Q = arrays.FloatArray(
        label=":math:`Q`",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the inhibitory population.
        Constant intensity.Entry point for coupling.""",
        order=14)

    a_e = arrays.FloatArray(
        label=":math:`a_e`",
        default=numpy.array([1.2]),
        range=basic.Range(lo=0.0, hi=1.4, step=0.01),
        doc="""The slope parameter for the excitatory response function""",
        order=15)

    a_i = arrays.FloatArray(
        label=":math:`a_i`",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.0, hi=2.0, step=0.01),
        doc="""The slope parameter for the inhibitory response function""",
        order=16)

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.0, 1.0]),
                 "I": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order=17)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["E", "I", "E + I", "E - I"],
        default=["E"],
        select_multiple=True,
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`
               and :math:`I = 1`.""",
        order=18)

    state_variables = 'E I'.split()
    _nvar = 2
    cvar = numpy.array([0, 1], dtype=numpy.int32)

    def __init__(self, num_nodes, simulation_length, dt, initial_values):
        self.number_of_nodes = num_nodes
        self.simulation_length = simulation_length
        self.num_iterations = int(simulation_length / dt)
        libmiindwc.init(num_nodes, simulation_length, dt)
        libmiindwc.setInitialValues(initial_values[0].tolist(), initial_values[1].tolist())

    def configure(self):
        """  """
        super(Miind_WilsonCowan, self).configure()
        self.update_derived_parameters()

        # Pass all parameters to the MIIND model
    	libmiindwc.initParams([self.c_ee[0], self.c_ei[0],
        self.c_ie[0], self.c_ii[0], self.tau_e[0], self.tau_i[0], self.k_e[0], self.k_i[0],
        self.r_e[0], self.r_i[0], self.a_e[0], self.a_i[0], self.b_e[0],
        self.b_i[0], self.P[0], self.Q[0]])
        libmiindwc.startSimulation()

    def dfun(self, x, c, local_coupling=0.0):
        E = x[0, :]
        I = x[1, :]

        # long-range coupling
        c_0 = c[0, :]
        c_1 = numpy.zeros(c_0.shape)

        # short-range (local) coupling
        lc_0 = local_coupling * E
        lc_1 = local_coupling * I

        coupling_E = c_0 + lc_0 + lc_1
        coupling_I = lc_0 + lc_1

        c_ = numpy.row_stack((coupling_E, coupling_I))[:,0]

    	x_ = numpy.array(libmiindwc.evolveSingleStep(c_.tolist()))

        return numpy.reshape(x_, x.shape)
