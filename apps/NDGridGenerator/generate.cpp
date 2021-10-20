#include <NdGrid.hpp>

void applyRybakInterneuronEuler(NdPoint& p, double t) {
    double g_nap = 0.25;
    double g_na = 30.0;
    double g_k = 6.0;
    double E_na = 55.0;
    double E_k = -80.0;
    double C = 1.0;
    double g_l = 0.1;
    double E_l = -64.0;
    double I = 3.4;
    double I_h = 0.0;

    double v = p.coords[2];
    double h_na = p.coords[1];
    double m_k = p.coords[0];

    for (unsigned int i = 0; i < 11; i++) {

        double I_l = -g_l * (v - E_l);
        double I_na = -g_na * h_na * (v - E_na) * (pow((pow((1 + exp(-(v + 35) / 7.8)), -1)), 3));
        double I_k = -g_k * (pow(m_k, 4)) * (v - E_k);

        double v_prime = I_na + I_k + I_l + I;

        double part_1 = (pow((1 + (exp((v + 55) / 7))), -1)) - h_na;
        double part_2 = 30 / (exp((v + 50) / 15) + exp(-(v + 50) / 16));
        double h_na_prime = part_1 / part_2;

        part_1 = (pow((1 + (exp(-(v + 28) / 15))), -1)) - m_k;
        part_2 = 7 / (exp((v + 40) / 40) + exp((-v + 40) / 50));
        double m_k_prime = part_1 / part_2;

        v = v + (t / 11.0) * v_prime;
        h_na = h_na + (t / 11.0) * h_na_prime;
        m_k = m_k + (t / 11.0) * m_k_prime;
    }

    p.coords[2] = v;
    p.coords[1] = h_na;
    p.coords[0] = m_k;

}

void applyHindmarshRoseEuler(NdPoint& p, double t) {
    double a = 1.0;
    double b = 3.0;
    double c = 1.0;
    double d = 5.0;
    double r = 0.002;
    double s = 4.0;
    double x_R = -1.6;
    double I = 3.14;
    double I_h = 0.0;

    double x = p.coords[2];
    double y = p.coords[1];
    double z = p.coords[0];

    for (unsigned int i = 0; i < 11; i++) {

        double x_prime = y + (-a * pow(x, 3) + b * pow(x, 2)) - z + I;
        double y_prime = c - (d * pow(x, 2)) - y;
        double z_prime = r * (s * (x - x_R) - z);

        x = x + (t / 11.0) * x_prime;
        y = y + (t / 11.0) * y_prime;
        z = z + (t / 11.0) * z_prime;
    }

    p.coords[2] = x;
    p.coords[1] = y;
    p.coords[0] = z;

}

void applyConductance3D(NdPoint& p, double t) {
    double tau_m = 20e-3;
    double E_r = -65e-3;
    double E_e = 0;
    double tau_s = 5e-3;
    double tau_t = 5e-3;
    double g_max = 0.8;
    double V_min = -66e-3;
    double V_max = -55e-3;
    double V_th = -55e-3;
    double N_V = 2000;
    double w_min = 0.0;
    double w_max = 10.0;
    double N_w = 20.0;
    double u_min = 0.0;
    double u_max = 10.0;
    double N_u = 20.0;

    double v = p.coords[2];
    double w = p.coords[1];
    double u = p.coords[0];

    for (unsigned int i = 0; i < 11; i++) {
        double v_prime = (-(v - E_r) - w * (v - E_e) - u * (v - E_e)) / tau_m;
        double w_prime = -w / tau_s;
        double u_prime = -u / tau_t;

        v = v + (t / 11.0) * v_prime;
        w = w + (t / 11.0) * w_prime;
        u = u + (t / 11.0) * u_prime;
    }

    p.coords[2] = v;
    p.coords[1] = w;
    p.coords[0] = u;
}

void applyConductance2D(NdPoint& p, double t) {
    double tau_m = 20e-3;
    double E_r = -65e-3;
    double E_e = 0;
    double tau_s = 5e-3;
    double tau_t = 5e-3;
    double g_max = 0.8;
    double V_min = -66e-3;
    double V_max = -55e-3;
    double V_th = -55e-3;
    double N_V = 2000;
    double w_min = 0.0;
    double w_max = 10.0;
    double N_w = 20.0;
    double u_min = 0.0;
    double u_max = 10.0;
    double N_u = 20.0;

    double v = p.coords[1];
    double w = p.coords[0];

    for (unsigned int i = 0; i < 11; i++) {
        double v_prime = (-(v - E_r) - w * (v - E_e)) / tau_m;
        double w_prime = -w / tau_s;

        v = v + (t / 11.0) * v_prime;
        w = w + (t / 11.0) * w_prime;
    }

    p.coords[1] = v;
    p.coords[0] = w;
}

void applyMauritzioExEuler(NdPoint& p, double t) {

    double C_m = 281.0;
    double g_L = 30.0;
    double t_ref = 2.0;
    double E_L = -70.6;
    double V_reset = -70.6;
    double E_ex = 0.0;
    double E_in = -75.0;
    double tau_syn_ex = 34.78;
    double tau_syn_in = 8.28;
    double a = 0.0;
    double b = 0.0;
    double Delta_T = 2.0;
    double tau_w = 0.0;
    double V_th = -50.4;
    double V_peak = -40.4;
    double I_e = 60.0;

    double V_m = p.coords[2];
    double g_e = p.coords[1];
    double g_i = p.coords[0];

    for (unsigned int i = 0; i < 100; i++) {
        if (V_m > -30.0)
            V_m = -30.0;

        double w = 0.0; // both tau_w and a are 0, so there is no adaptation.
        double V_m_prime = (-(g_L * (V_m - E_L)) + g_L * Delta_T * exp((V_m - V_th) / Delta_T) - (g_e * (V_m - E_ex)) - (g_i * (V_m - E_in)) - w + I_e) / C_m;
        double g_e_prime = (-g_e / tau_syn_ex);// + (2870.249*0.001*0.1);
        double g_i_prime = (-g_i / tau_syn_in);// + (245.329*0.001*1.2671875);

        V_m = V_m + (t / 100.0) * V_m_prime;
        g_e = g_e + (t / 100.0) * g_e_prime;
        g_i = g_i + (t / 100.0) * g_i_prime;
    }

    p.coords[2] = V_m;
    p.coords[1] = g_e;
    p.coords[0] = g_i;

}

void applyMauritzioInEuler(NdPoint& p, double t) {

    double C_m = 281.0;
    double g_L = 30.0;
    double t_ref = 1.0;
    double E_L = -70.6;
    double V_reset = -70.6;
    double E_ex = 0.0;
    double E_in = -75.0;
    double tau_syn_ex = 26.55;
    double tau_syn_in = 8.28;
    double a = 0.0;
    double b = 0.0;
    double Delta_T = 2.0;
    double tau_w = 0.0;
    double V_th = -50.4;
    double V_peak = -40.4;
    double I_e = 0.0;

    double V_m = p.coords[2];
    double g_e = p.coords[1];
    double g_i = p.coords[0];

    for (unsigned int i = 0; i < 100; i++) {
        if (V_m > -30.0)
            V_m = -30.0;

        double w = 0.0; // both tau_w and a are 0, so there is no adaptation.
        double V_m_prime = (-(g_L * (V_m - E_L)) + g_L * Delta_T * exp((V_m - V_th) / Delta_T) - (g_e * (V_m - E_ex)) - (g_i * (V_m - E_in)) - w + I_e) / C_m;
        double g_e_prime = (-g_e / tau_syn_ex);// + (733.4874*0.001*0.4875);
        double g_i_prime = (-g_i / tau_syn_in);// + (86.1246*0.001*1.2671875);

        V_m = V_m + (t / 100.0) * V_m_prime;
        g_e = g_e + (t / 100.0) * g_e_prime;
        g_i = g_i + (t / 100.0) * g_i_prime;
    }

    p.coords[2] = V_m;
    p.coords[1] = g_e;
    p.coords[0] = g_i;

}

int main() {
	std::vector<double> base = { -0.2,-66e-3};
	std::vector<double> dims = { 2.2, 12e-3};
	std::vector<unsigned int> res = {50,100};
	double threshold = -55e-3;
	double reset_v = -65e-3;
    NdGrid g(base, dims, res, threshold, reset_v, 1e-05);

    g.setCppFunction(applyConductance2D);
	g.generateModelFile("conductanceNdNoise", 1);
	g.generateTMatFileBatched("conductanceNdNoise");
	return 0;
}
