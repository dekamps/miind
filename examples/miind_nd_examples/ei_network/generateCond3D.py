import miind.grid_generate as grid_generate
import miind.miindgen as miindgen

# model parameters taken from "Adaptive exponential integrate-and-fire model as an effective description of neuronal activity" and "Fluctuating synaptic conductances recreate in vivo-like activity in neocortical neurons"
def cond(y):
    E_l = -70.6
    V_thres = -50.4 
    E_e = 0.0
    E_i = -75
    C = 281
    g_l = 0.03
    tau_e = 2.728
    tau_i = 10.49

    v = y[2]
    w = y[1]
    u = y[0]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (v - E_i)) / C
    w_prime = -(w) / tau_e
    u_prime = -(u) / tau_i

    return [u_prime, w_prime, v_prime]

miindgen.generateNdGrid(cond, 'cond3d_100x10x40', [-5,-1,-80], [55,11,40], [100,100,150], -50.4, -70.6, [0.0,0.0,0.0], 1, 0.001)
miindgen.generateNdGrid(cond, 'cond3d_100x25x40', [-5,-1,-80], [55,26,40], [100,100,150], -50.4, -70.6, [0.0,0.0,0.0], 1, 0.001)