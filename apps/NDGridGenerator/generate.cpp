#include "Grid.hpp"

int main() {
	std::vector<double> base = {-0.2,-0.2,-66e-3};
	std::vector<double> dims = {2.2,2.2, 12e-3};
	std::vector<unsigned int> res = {50,100,100};
	double threshold = -55e-3;
	double reset_v = -65e-3;
	Grid g(base, dims, res, threshold, reset_v, 1e-05);

	g.generateModelFile("conductanceNdNoise", 1);
	g.generateTMatFileBatched("conductanceNdNoise");
	return 0;
}
