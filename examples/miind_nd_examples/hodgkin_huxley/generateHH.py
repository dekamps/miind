import numpy as np
import miind.miindgen as miindgen

def hodgkin_huxley(y):
    V_k = -90
    V_na = 50
    V_l = -65
    g_k = 30
    g_na = 100
    g_l = 0.5
    C = 1.0
    V_t = -63

    v = y[3];
    m = y[2];
    n = y[1];
    h = y[0];

    alpha_m = (0.32 * (13 - v + V_t)) / (np.exp((13 - v + V_t) / 4) - 1)
    alpha_n = (0.032 * (15 - v + V_t)) / (np.exp((15 - v + V_t) / 5) - 1)
    alpha_h = 0.128 * np.exp((17 - v + V_t) / 18)

    beta_m = (0.28 * (v - V_t - 40)) / (np.exp((v - V_t - 40) / 5) - 1)
    beta_n = 0.5 * np.exp((10 - v + V_t) / 40)
    beta_h = 4 / (1 + np.exp((40 - v + V_t) / 5))

    v_prime = (-(g_k * n**4 * (v - V_k)) - (g_na * m**3 * h * (v - V_na)) - (g_l * (v - V_l))) / C
    m_prime = (alpha_m * (1 - m)) - (beta_m * m)
    n_prime = (alpha_n * (1 - n)) - (beta_n * n)
    h_prime = (alpha_h * (1 - h)) - (beta_h * h)

    return [h_prime, n_prime, m_prime, v_prime]

miindgen.generateNdGrid(hodgkin_huxley, "hh_50x50x50x50", [ -0.1, -0.1, -0.1, -100], [1.2, 1.2, 1.2, 160], [50,50,50,50], 59.9, -99.9, [0.0,0.0,0.0], 0.01, 0.001)