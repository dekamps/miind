#include "Grid.hpp"

int main() {
	std::vector<double> base = {-2.0,-2.0,-75};
	std::vector<double> dims = {60.0,60.0,40};
	std::vector<unsigned int> res = {30,30,100};
	double threshold = -40.3;
	double reset_v = -70.6;
	Grid g(base, dims, res, threshold, reset_v, 0.01);

	g.generateModelFile("conductanceNdNoise", 0.001);
	g.generateTMatFileBatched("conductanceNdNoise");
	return 0;
}
