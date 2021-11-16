import miind.miindgen as miindgen
    
def tsodyks(y):
    tau_intact = 3 #ms
    tau_rec = 450 #ms
    U_se = 0.55
    A_se = 530 #pA
    
    R_in = 100 #MOhms
    
    V_r = -65 #mV
    tau_mem = 30 #ms

    V = y[2];
    E = y[1]; # synaptic facilitation / Release probability
    R = y[0]; # synaptic depression
    
    V_prime = (-(v - V_r) - (R_in * v * A_se * e)) / tau_mem
    E_prime = -e / tau_intact
    R_prime = (1 - r - e) / tau_rec

    return [R_prime, E_prime, V_prime]

miindgen.generateNdGrid(tsodyks, "synapse", [-0.2,-0.2,-70], [1.4, 1.4, 40], [50,100,50], -35, -65, [0.0,0.0,0.0], 0.01, 1)