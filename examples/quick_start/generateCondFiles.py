import miind.grid_generate as grid_generate

def cond(y,t):
    E_r = -65e-3
    tau_m = 20e-3
    tau_s = 5e-3

    v = y[0];
    h = y[1];

    v_prime = ( -(v - E_r) - (h * v) ) / tau_m
    h_prime = -h / tau_s

    return [v_prime, h_prime]

grid_generate.generate(
    func = cond, 
    timestep = 1e-05, 
    timescale = 1, 
    tolerance = 1e-6, 
    basename = 'cond', 
    threshold_v = -55.0e-3, 
    reset_v = -65e-3, 
    reset_shift_h = 0.0, 
    grid_v_min = -72.0e-3, 
    grid_v_max = -54.0e-3, 
    grid_h_min = -1.0, 
    grid_h_max = 2.0, 
    grid_v_res = 200, 
    grid_h_res = 200, 
    efficacy_orientation = 'h')