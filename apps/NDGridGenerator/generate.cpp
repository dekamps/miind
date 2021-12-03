#include <NdGrid.hpp>

// Recommended settings in main() for applyHindmarshRoseEuler:
// std::vector<double> base = { 0,-30,-5 };
// std::vector<double> dims = { 8, 40, 10 };
// std::vector<unsigned int> res = { 50, 50,50 };
// std::vector<double> reset_relative = { 0.0,0.0,0.0 };
// double threshold = 4.99;
// double reset_v = -4.99;
// NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.1);
//
// g.setCppFunction(applyHindmarshRoseEuler);
// g.generateModelFile("HindmarshRose", 0.001);
// g.generateTMatFileBatched("HindmarshRose");
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

// Recommended settings in main() for applyConductance3D:
// std::vector<double> base = { -0.2,-0.2,-66e-3 };
// std::vector<double> dims = { 2.2, 2.2, 12e-3 };
// std::vector<unsigned int> res = { 50, 50,100 };
// std::vector<double> reset_relative = { 0.0,0.0,0.0 };
// double threshold = -55e-3;
// double reset_v = -65e-3;
// NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-05);
//
// g.setCppFunction(applyConductance3D);
// g.generateModelFile("conductanceNdNoise", 1);
// g.generateTMatFileBatched("conductanceNdNoise");
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

// Recommended settings in main() for applyConductance2D:
// std::vector<double> base = { -0.2,-66e-3 };
// std::vector<double> dims = { 2.2, 12e-3 };
// std::vector<unsigned int> res = { 100,100 };
// std::vector<double> reset_relative = { 0.0,0.0 };
// double threshold = -55e-3;
// double reset_v = -65e-3;
// NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-05);
//
// g.setCppFunction(applyConductance2D);
// g.generateModelFile("conductanceNdNoise", 1);
// g.generateTMatFileBatched("conductanceNdNoise");
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

// Recommended settings in main() for applyTsodyks:
// std::vector<double> base = { -0.2,-0.2,-66 };
// std::vector<double> dims = { 1.4, 1.4, 12 };
// std::vector<unsigned int> res = { 50, 50, 50 };
// std::vector<double> reset_relative = { 0.0,0.0,0.0 };
// double threshold = -55;
// double reset_v = -65;
// NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-02);
//
// g.setCppFunction(applyTsodyks);
// g.generateModelFile("synapse", 1);
// g.generateTMatFileBatched("synapse");
void applyTsodyks(NdPoint& p, double t) {
    double tau_intact = 3;
    double tau_rec = 700;
    double U_se = 0.65;
    double A_se = 250; //250 for Hi, 500 for Lo
    double R_in = 0.02;
    double V_r = -65;
    double tau_mem = 25; 

    double v = p.coords[2];
    double e = p.coords[1];
    double r = p.coords[0];

    for (unsigned int i = 0; i < 11; i++) {
        double v_prime = (-(v - V_r) - (R_in * v * A_se * e)) / tau_mem;
        double e_prime = -(e / tau_intact);
        double r_prime = ((1 - r - std::abs(e)) / tau_rec);

        v = v + (t / 11.0) * v_prime;
        e = e + (t / 11.0) * e_prime;
        r = r + (t / 11.0) * r_prime;
    }

    p.coords[2] = v;
    p.coords[1] = e;
    p.coords[0] = r;
}

// Recommended settings in main() for applyBRMNRedux:
// std::vector<double> base = { -0.2, -1.5, -1.5 };
//std::vector<double> dims = { 1.4, 3.0, 3.0 };
//std::vector<unsigned int> res = { 50, 50, 50 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0 };
//double threshold = 0.9;
//double reset_v = -0.5;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.0001);
//
//g.setCppFunction(applyBRMNRedux);
//g.generateModelFile("brmn", 1);
//g.generateTMatFileBatched("brmn");
void applyBRMNRedux(NdPoint& p, double t) {
    double V_ca = 1.0;
    double V_k = -0.7;

    double g_na = 0.01;
    double g_ca = 1.5;
    double g_k = 0.5;
    double g_l = 0.5;
    double V_l = -0.5;

    double v_1 = 0.05;
    double v_2 = 0.15;
    double v_3 = 0;
    double v_4 = 0.1;

    double phi_d = 0.2;

    double tau_v_s = 1.0;
    double tau_v_d = 1.0;
    double tau_w_d = 1.0;

    double g_c = 0.25;
    double pp = 0.5;

    double v_s = p.coords[2];
    double v_d = p.coords[1];
    double w_d = p.coords[0];
    

    for (unsigned int i = 0; i < 11; i++) {
        double I_l = -g_l * (v_d - V_l);
        double I_ca = -g_ca * (v_d - V_ca) * ((1.0 / 2.0) * (1.0 + (tanh((v_d - v_1) / v_2))));
        double I_k = -g_k * w_d * (v_d - V_k);

        double w_inf = ((1.0 / 2.0) * (1.0 + (tanh((v_d - v_3) / v_4))));
        double tau_s = (1.0 / cosh((v_d - v_3) / (2.0 * v_4)));

        double v_s_prime = (-(v_s - V_l) + ((g_c / pp) * (v_d - v_s))) / tau_v_s; // Soma is just a LIF
        double v_d_prime = (I_l + I_ca + I_k + ((g_c / (1.0 - pp)) * (v_s - v_d))) / tau_v_d;
        double w_d_prime = (phi_d * (w_inf - w_d) / tau_s) / tau_w_d;

        v_s = v_s + (t / 11.0) * v_s_prime;
        v_d = v_d + (t / 11.0) * v_d_prime;
        w_d = w_d + (t / 11.0) * w_d_prime;

    }

    p.coords[2] = v_s;
    p.coords[1] = v_d;
    p.coords[0] = w_d;
}

int main() {
    std::vector<double> base = { -0.2, -1.5, -1.5 };
    std::vector<double> dims = { 1.4, 3.0, 3.0 };
    std::vector<unsigned int> res = { 50, 50, 50 };
    std::vector<double> reset_relative = { 0.0,0.0,0.0 };
    double threshold = 0.4;
    double reset_v = -0.5;
    NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.1);
 
    g.setCppFunction(applyBRMNRedux);
    g.generateModelFile("brmn", 0.001);
    g.generateTMatFileBatched("brmn");
	return 0;
}
