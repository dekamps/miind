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
    
    V_prime = ( -(V - V_r) - (R_in * E * V)) / tau_mem
    E_prime = - (E / tau_intact)# + U_se*R
    R_prime = ((1 - R - E) / tau_rec)# - U_se*R
    

    return [R_prime, E_prime, V_prime]

miindgen.generateNdGrid(tsodyks, "synapse", [-0.2,-0.2,-66], [1.4, 1.4, 12], [50,50,50], -55, -65, [0.0,0.0,0.0], 0.01, 1)