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
    double tau_rec = 800;
    double U_se = 0.67;
    double A_se = 250;
    double R_in = 100;
    double V_r = -65;
    double tau_mem = 50;

    double v = p.coords[2];
    double e = p.coords[1];
    double r = p.coords[0];

    for (unsigned int i = 0; i < 11; i++) {
        double v_prime = (-(v - V_r) - (R_in * e * v)) / tau_mem;
        double e_prime = -(e / tau_intact);
        double r_prime = ((1 - r - e) / tau_rec);

        v = v + (t / 11.0) * v_prime;
        e = e + (t / 11.0) * e_prime;
        r = r + (t / 11.0) * r_prime;
    }

    p.coords[2] = v;
    p.coords[1] = e;
    p.coords[0] = r;
}

int main() {
     std::vector<double> base = { -0.2,-0.2,-66 };
     std::vector<double> dims = { 1.4, 1.4, 100 };
     std::vector<unsigned int> res = { 50, 50, 50 };
     std::vector<double> reset_relative = { 0.0,0.0,0.0 };
     double threshold = 30;
     double reset_v = -65;
     NdGrid g(base, dims, res, threshold, reset_v, reset_relative, 1e-02);

     g.setCppFunction(applyTsodyks);
     g.generateModelFile("synapse50", 0.001);
     g.generateTMatFileBatched("synapse50");
	return 0;
}
