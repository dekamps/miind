#include <NdGrid.hpp>
#include <cmath>

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
//std::vector<double> base = { -0.2,-0.2,-90e-3 };
//std::vector<double> dims = { 1.4, 1.4, 36e-3 };
//std::vector<unsigned int> res = { 50, 50, 100 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0 };
//double threshold = -55e-3;
//double reset_v = -65e-3;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-04);
//
//g.setCppFunction(applyConductance3D);
//g.generateModelFile("conductanceNdNoise", 1);
//g.generateTMatFileBatched("conductanceNdNoise");
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
        double v_prime = (-(v - E_r) - w * (v - E_e) + u * (v - E_e)) / tau_m;
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
//std::vector<double> base = { -0.2,-1.2,-100 };
//std::vector<double> dims = { 1.4, 2.4, 60 };
//std::vector<unsigned int> res = { 50, 50, 50 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0 };
//double threshold = -45;
//double reset_v = -65;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-01);
//
//g.setCppFunction(applyTsodyks);
//g.generateModelFile("synapse", 0.001);
//g.generateTMatFileBatched("synapse");
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

// Recommended settings in main() for applyTsodyks4d:
//std::vector<double> base = { -0.2,-0.2,-10, -2 };
//std::vector<double> dims = { 1.4, 1.4, 120, 5 };
//std::vector<unsigned int> res = { 50, 50, 50, 50 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0,0.0 };
//double threshold = 2.9;
//double reset_v = -1.9;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.1);
//
//g.setCppFunction(applyTsodyks4d);
//g.generateModelFile("synapse", 0.001);
//g.generateTMatFileBatched("synapse");
void applyTsodyks4d(NdPoint& p, double t) {
    double tau_intact = 3;
    double tau_rec = 250;
    double U_se = 0.67;
    double A_se = 250; //250 for Hi, 500 for Lo
    double tau_mem = 50;

    double v = p.coords[3];
    double u = p.coords[2];
    double e = p.coords[1];
    double r = p.coords[0];

    for (unsigned int i = 0; i < 11; i++) {
        double v_prime = (-v + u) / tau_mem;
        double u_prime = -u;
        double e_prime = -(e / tau_intact);
        double r_prime = ((1 - r - std::abs(e)) / tau_rec);

        v = v + (t / 11.0) * v_prime;
        u = u + (t / 11.0) * u_prime;
        e = e + (t / 11.0) * e_prime;
        r = r + (t / 11.0) * r_prime;
    }

    p.coords[3] = v;
    p.coords[2] = u;
    p.coords[1] = e;
    p.coords[0] = r;
}

// Recommended settings in main() for applyBRMNRedux:
//std::vector<double> base = { -0.2, -1.5, -2.5 };
//std::vector<double> dims = { 1.4, 3.0, 3.5 };
//std::vector<unsigned int> res = { 50, 50, 100 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0 };
//double threshold = 0.5;
//double reset_v = -0.5;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.01);
//
//g.setCppFunction(applyBRMNRedux);
//g.generateModelFile("brmn", 0.01);
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

        double v_s_exp_prime = ((0.3 * 1.48 * std::exp((v_s - 0.5) / 1.48)) - (0.3 * (v_s + 1.0)) + ((g_c / pp) * (v_d - v_s))) / tau_v_s;

        double v_s_prime = (-(v_s - V_l) + ((g_c / pp) * (v_d - v_s))) / tau_v_s; // Soma is just a LIF
        double v_d_prime = (I_l + I_ca + I_k + ((g_c / (1.0 - pp)) * (v_s - v_d))) / tau_v_d;
        double w_d_prime = (phi_d * (w_inf - w_d) / tau_s) / tau_w_d;

        v_s = v_s + (t / 11.0) * v_s_exp_prime;
        v_d = v_d + (t / 11.0) * v_d_prime;
        w_d = w_d + (t / 11.0) * w_d_prime;

    }

    p.coords[2] = v_s;
    p.coords[1] = v_d;
    p.coords[0] = w_d;
}

// Recommended settings in main() for applyBRMN:
//std::vector<double> base = { -0.2, -0.2, -1.5, -2.5 };
//std::vector<double> dims = { 1.4, 1.4, 3.0, 4.5 };
//std::vector<unsigned int> res = { 50, 50, 50, 50 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0,0.0 };
//double threshold = 1.99;
//double reset_v = -2.49;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.01);
//
//g.setCppFunction(applyBRMN);
//g.generateModelFile("brmn4d", 0.001);
//g.generateTMatFileBatched("brmn4d");
void applyBRMN(NdPoint& p, double t) {

    double s_V_na = 1;
    double s_V_k = -0.7;

    double s_g_na = 1.0;
    double s_g_kdr = 2.0;
    double s_g_l = 0.5;
    double s_V_l = -0.5;

    double s_v_1 = -0.01;
    double s_v_2 = 0.15;
    double s_v_3 = -0.04;
    double s_v_4 = 0.1;

    double s_phi_s = 0.2;

    double d_V_ca = 1.0;
    double d_V_k = -0.7;

    double d_g_na = 0.01;
    double d_g_ca = 1.5;
    double d_g_k = 0.5;
    double d_g_l = 0.5;
    double d_V_l = -0.5;

    double d_v_1 = 0.05;
    double d_v_2 = 0.15;
    double d_v_3 = 0;
    double d_v_4 = 0.1;

    double d_phi_d = 0.2;

    double tau_v_s = 1.0;
    double tau_v_d = 1.0;
    double tau_w_d = 1.0;
    double tau_w_s = 1.0;

    double g_c = 0.25;
    double pp = 0.5;

    double v_s = p.coords[3];
    double v_d = p.coords[2];
    double w_d = p.coords[1];
    double w_s = p.coords[0];


    for (unsigned int i = 0; i < 1; i++) {
        double d_I_l = -d_g_l * (v_d - d_V_l);
        double d_I_ca = -d_g_ca * (v_d - d_V_ca) * ((1.0 / 2.0) * (1.0 + (tanh((v_d - d_v_1) / d_v_2))));
        double d_I_k = -d_g_k * w_d * (v_d - d_V_k);

        double d_w_inf = ((1.0 / 2.0) * (1.0 + (tanh((v_d - d_v_3) / d_v_4))));
        double d_tau_s = (1.0 / cosh((v_d - d_v_3) / (2.0 * d_v_4)));

        double s_I_l = -s_g_l * (v_s - s_V_l);
        double s_I_na = -s_g_na * (v_s - s_V_na) * ((1 / 2) * (1 + (tanh((v_s - s_v_1) / s_v_2))));
        double s_I_k = -s_g_kdr * w_d * (v_s - s_V_k);

        double s_w_inf = ((1 / 2) * (1 + (tanh((v_s - s_v_3) / s_v_4))));
        double s_tau_s = (1 / cosh((v_s - s_v_3) / (2 * s_v_4)));

        double v_s_prime = (s_I_l + s_I_na + s_I_k + ((g_c / pp) * (v_d - v_s))) / tau_v_s;
        double w_s_prime = (s_phi_s * (s_w_inf - w_d) / s_tau_s) / tau_w_s;
        double v_d_prime = (d_I_l + d_I_ca + d_I_k + ((g_c / (1.0 - pp)) * (v_s - v_d))) / tau_v_d;
        double w_d_prime = (d_phi_d * (d_w_inf - w_d) / d_tau_s) / tau_w_d;

        v_s = v_s + (t / 1.0) * v_s_prime;
        v_d = v_d + (t / 1.0) * v_d_prime;
        w_d = w_d + (t / 1.0) * w_d_prime;
        w_s = w_s + (t / 1.0) * w_s_prime;

    }

    p.coords[3] = v_s;
    p.coords[2] = v_d;
    p.coords[1] = w_d;
    p.coords[0] = w_s;
}

// Recommended settings in main() for applyRinzelBurster:
//std::vector<double> base = { -0.2,-0.2,-100 };
//std::vector<double> dims = { 1.2, 1.2, 160 };
//std::vector<unsigned int> res = { 50, 50, 100 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0 };
//double threshold = 59.9;
//double reset_v = -79.9;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-02);
//
//g.setCppFunction(applyRinzelBurster);
//g.generateModelFile("rinzel", 0.001);
//g.generateTMatFileBatched("rinzel");
void applyRinzelBurster(NdPoint& p, double t) {

    double g_nap = 0.25;
    double g_na = 30;
    double g_k = 6;
    double theta_m = -47.1; //mV
    double sig_m = -3.1; //mV
    double theta_h = -59; //mV
    double sig_h = 8; //mV
    double tau_h = 1200; //ms
    double E_na = 55; //mV
    double E_k = -80;
    double C = 1; //uF
    double g_l = 0.1; //mS
    double E_l = -64.0; //mV

    double v = p.coords[2];
    double h_nap = p.coords[1];
    double h_na = p.coords[0];
    double m_k = (0.033 / (h_na + 0.06));

    double I = 0.0;

    for (unsigned int i = 0; i < 1; i++) {

        double I_nap = -g_nap * h_nap * (v - E_na) * (1.0 / (1 + exp(-(v + 47.1) / 3.1)));
        double I_l = -g_l * (v - E_l);
        double I_na = -g_na * h_na * (v - E_na) * pow((1.0 / (1 + exp(-(v + 35) / 7.8))), 3);
        double I_k = -g_k * (pow(m_k, 4)) * (v - E_k);

        double v_prime = I_nap + I_na + I_k + I_l + I;

        double part_1 = (1.0 / (1 + (exp((v + 59) / 8)))) - h_nap;
        double part_2 = (1200 / cosh((v + 59) / (16)));
        double h_nap_prime = part_1 / part_2;

        part_1 = (1.0 / (1 + (exp((v + 55) / 7)))) - h_na;
        part_2 = 30 / (exp((v + 50) / 15) + exp(-(v + 50) / 16));
        double h_na_prime = part_1 / part_2;

        v = v + (t / 1.0) * v_prime;
        h_nap = h_nap + (t / 1.0) * h_nap_prime;
        h_na = h_na + (t / 1.0) * h_na_prime;
    }

    p.coords[2] = v;
    p.coords[1] = h_nap;
    p.coords[0] = h_na;
}

// Recommended settings in main() for applyRinzelBurster:
//std::vector<double> base = { -0.2,-130 };
//std::vector<double> dims = { 1.2, 100 };
//std::vector<unsigned int> res = { 200, 200 };
//std::vector<double> reset_relative = { -0.006, 0.0 };
//double threshold = -36.0;
//double reset_v = -52.0;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-01);
//
//g.setCppFunction(applyRinzelBurster2d);
//g.generateModelFile("rinzel2D", 0.001);
//g.generateTMatFileBatched("rinzel2D");
void applyRinzelBurster2d(NdPoint& p, double t) {

    double g_nap = 0.25;
    double g_na = 30;
    double g_k = 11.2;
    double theta_m = -47.1; //mV
    double sig_m = -3.1; //mV
    double theta_h = -59; //mV
    double sig_h = 8; //mV
    double tau_h = 1200; //ms
    double E_na = 55; //mV
    double E_k = -80;
    double C = 1; //uF
    double g_l = 0.1; //mS
    double E_l = -64.0; //mV

    double v = p.coords[1];
    double h_nap = p.coords[0];

    double h_na = 0;//#(1.0 / (1 + (exp((v + 55) / 7)))); //0.62
    double m_k = 0;//#(1.0 / (1 + (exp(-(v + 28) / 15))));

    double I = 0.0;

    for (unsigned int i = 0; i < 1; i++) {

        double I_nap = -g_nap * h_nap * (v - E_na) * (1.0 / (1 + exp(-(v + 47.1) / 3.1)));
        double I_l = -g_l * (v - E_l);
        double I_na = -g_na * h_na * (v - E_na) * pow((1.0 / (1 + exp(-(v + 35) / 7.8))),3);
        double I_k = -g_k * pow((m_k),4) * (v - E_k);

        double v_prime = I_nap + I_na + I_k + I_l + I;

        double part_1 = (1.0 / (1 + (exp((v + 59) / 8)))) - h_nap;
        double part_2 = (1200 / cosh((v + 59) / (16)));
        double h_nap_prime = part_1 / part_2;

        v = v + (t / 1.0) * v_prime;
        h_nap = h_nap + (t / 1.0) * h_nap_prime;
    }

    p.coords[1] = v;
    p.coords[0] = h_nap;
}

// Recommended settings in main() for applyHH:
//std::vector<double> base = { -0.1, -0.1, -0.1, -100 };
//std::vector<double> dims = { 1.2, 1.2, 1.2, 160 };
//std::vector<unsigned int> res = { 40, 40, 40, 100 };
//std::vector<double> reset_relative = { 0.0,0.0,0.0,0.0 };
//double threshold = 59.9;
//double reset_v = -99.9;
//NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.01);
//
//g.setCppFunction(applyHH);
//g.generateModelFile("hh", 0.001);
//g.generateTMatFileBatched("hh");
void applyHH(NdPoint& p, double t) {
    double V_k = -90;
    double V_na = 50;
    double V_l = -65;
    double g_k = 30;
    double g_na = 100;
    double g_l = 0.5;
    double C = 1.0;
    double V_t = -63;

    double v = p.coords[3];
    double m = p.coords[2];
    double n = p.coords[1];
    double h = p.coords[0];

    for (unsigned int i = 0; i < 1; i++) {

        double alpha_m = (0.32 * (13 - v + V_t)) / (exp((13 - v + V_t) / 4) - 1);
        double alpha_n = (0.032 * (15 - v + V_t)) / (exp((15 - v + V_t) / 5) - 1);
        double alpha_h = 0.128 * exp((17 - v + V_t) / 18);

        double beta_m = (0.28 * (v - V_t - 40)) / (exp((v - V_t - 40) / 5) - 1);
        double beta_n = 0.5 * exp((10 - v + V_t) / 40);
        double beta_h = 4 / (1 + exp((40 - v + V_t) / 5));

        double v_prime = (-(g_k * pow(n,4) * (v - V_k)) - (g_na * pow(m,3) * h * (v - V_na)) - (g_l * (v - V_l))) / C;
        double m_prime = (alpha_m * (1 - m)) - (beta_m * m);
        double n_prime = (alpha_n * (1 - n)) - (beta_n * n);
        double h_prime = (alpha_h * (1 - h)) - (beta_h * h);

        v = v + (t / 1.0) * v_prime;
        m = m + (t / 1.0) * m_prime;
        n = n + (t / 1.0) * n_prime;
        h = h + (t / 1.0) * h_prime;
    }

    p.coords[3] = v;
    p.coords[2] = m;
    p.coords[1] = n;
    p.coords[0] = h;
}

int main() {
    std::vector<double> base = { -0.2, -0.2, -1.5, -2.5 };
    std::vector<double> dims = { 1.4, 1.4, 3.0, 4.5 };
    std::vector<unsigned int> res = { 50, 50, 50, 50 };
    std::vector<double> reset_relative = { 0.0,0.0,0.0,0.0 };
    double threshold = 1.99;
    double reset_v = -2.49;
    NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 0.01);
    
    g.setCppFunction(applyBRMN);
    g.generateModelFile("brmn4d", 0.001);
    g.generateTMatFileBatched("brmn4d");
	return 0;
}
