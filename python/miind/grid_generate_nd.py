import miind.miindgen as miindgen

def cond(y):
    E_r = -65e-3
    tau_m = 20e-3
    tau_s = 5e-3

    v = y[2];
    w = y[1];
    u = y[0];

    v_prime = ( -(v - E_r) - (w * v) - (u * v)) / tau_m
    w_prime = -w / tau_s
    u_prime = -u / tau_s

    return [u_prime, w_prime, v_prime]

miindgen.generateNdGrid(cond, "cond3D", [-0.2,-0.2,-66e-3], [2.2,2.2, 12e-3], [50,50,100], -55e-3, -65e-3, 1e-05, 1)