import pytest
import os
import json
import copy
import xml.etree.ElementTree as ET


def test_modelfiles():
    from miindio import MiindIO
    io = MiindIO(pytest.XML_PATH)
    assert io.modelfiles == ["aexp.model", "aexpnoa.model"]


def test_marginal_density(io_run):
    from numpy import trapz
    v1 = io_run.density['aexp']['v']
    bins_v1 = io_run.density['aexp']['bins_v']
    dv1 = bins_v1[1] - bins_v1[0]
    w1 = io_run.density['aexp']['w']
    bins_w1 = io_run.density['aexp']['bins_w']
    dw1 = bins_w1[1] - bins_w1[0]
    v2 = io_run.density['aexpnoa']['v']
    bins_v2 = io_run.density['aexpnoa']['bins_v']
    dv2 = bins_v2[1] - bins_v2[0]
    w2 = io_run.density['aexpnoa']['w']
    bins_w2 = io_run.density['aexpnoa']['bins_w']
    dw2 = bins_w2[1] - bins_w2[0]
    TOL = 0.1
    assert all(trapz(v1, dx=dv1, axis=1) - 1 < TOL)
    assert all(trapz(w1, dx=dw1, axis=1) - 1 < TOL)
    assert all(trapz(v2, dx=dv2, axis=1) - 1 < TOL)
    assert all(trapz(w2, dx=dw2, axis=1) - 1 < TOL)


def test_plot_marginal_density(io_run):
    io_run.density.plot_marginal_density()
