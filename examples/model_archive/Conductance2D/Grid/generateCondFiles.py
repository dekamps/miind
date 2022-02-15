import miind.grid_generate as grid_generate
import miind.miindgen as miindgen

def cond(y):
    E_r = -65e-3
    tau_m = 20e-3
    tau_s = 5e-3

    v = y[1];
    h = y[0];

    v_prime = ( -(v - E_r) - (h * v) ) / tau_m
    h_prime = -h / tau_s

    return [h_prime, v_prime]

miindgen.generateNdGrid(cond, 'cond', [-1.0,-72.0e-3], [3.0,22.0e-3], [200,200], -55.0e-3, -65e-3, [0.0,0.0], 1e-05, 1)