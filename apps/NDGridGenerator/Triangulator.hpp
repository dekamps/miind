#ifndef APP_ND_GRID_TRIANGULATOR
#define APP_ND_GRID_TRIANGULATOR

#include <map>
#include <vector>


class Point;
class Simplex;

class Triangulator {
public:
	std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>> transitions;
	Triangulator() {
		transitions = std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>>();
		transitions[1] = std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>();
		transitions[1][3] = std::map<unsigned int, std::vector<std::vector<unsigned int>>>();
		transitions[1][3][3] = std::vector<std::vector<unsigned int>>(4);
		transitions[1][3][3][0] = std::vector<unsigned int>(4);
		transitions[1][3][3][0][0] = 6;
		transitions[1][3][3][0][1] = 1;
		transitions[1][3][3][0][2] = 2;
		transitions[1][3][3][0][3] = 3;
		transitions[1][3][3][1] = std::vector<unsigned int>(4);
		transitions[1][3][3][1][0] = 5;
		transitions[1][3][3][1][1] = 6;
		transitions[1][3][3][1][2] = 1;
		transitions[1][3][3][1][3] = 2;
		transitions[1][3][3][2] = std::vector<unsigned int>(4);
		transitions[1][3][3][2][0] = 4;
		transitions[1][3][3][2][1] = 5;
		transitions[1][3][3][2][2] = 6;
		transitions[1][3][3][2][3] = 0;
		transitions[1][3][3][3] = std::vector<unsigned int>(4);
		transitions[1][3][3][3][0] = 4;
		transitions[1][3][3][3][1] = 5;
		transitions[1][3][3][3][2] = 6;
		transitions[1][3][3][3][3] = 1;
		transitions[1][2] = std::map<unsigned int, std::vector<std::vector<unsigned int>>>();
		transitions[1][2][3] = std::vector<std::vector<unsigned int>>(3);
		transitions[1][2][3][0] = std::vector<unsigned int>(4);
		transitions[1][2][3][0][0] = 4;
		transitions[1][2][3][0][1] = 1;
		transitions[1][2][3][0][2] = 2;
		transitions[1][2][3][0][3] = 5;
		transitions[1][2][3][1] = std::vector<unsigned int>(4);
		transitions[1][2][3][1][0] = 3;
		transitions[1][2][3][1][1] = 4;
		transitions[1][2][3][1][2] = 5;
		transitions[1][2][3][1][3] = 0;
		transitions[1][2][3][2] = std::vector<unsigned int>(4);
		transitions[1][2][3][2][0] = 3;
		transitions[1][2][3][2][1] = 4;
		transitions[1][2][3][2][2] = 1;
		transitions[1][2][3][2][3] = 5;
		transitions[1][1] = std::map<unsigned int, std::vector<std::vector<unsigned int>>>();
		transitions[1][1][3] = std::vector<std::vector<unsigned int>>(2);
		transitions[1][1][3][0] = std::vector<unsigned int>(4);
		transitions[1][1][3][0][0] = 2;
		transitions[1][1][3][0][1] = 3;
		transitions[1][1][3][0][2] = 4;
		transitions[1][1][3][0][3] = 0;
		transitions[1][1][3][1] = std::vector<unsigned int>(4);
		transitions[1][1][3][1][0] = 2;
		transitions[1][1][3][1][1] = 1;
		transitions[1][1][3][1][2] = 3;
		transitions[1][1][3][1][3] = 4;
		transitions[2] = std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>();
		transitions[2][2] = std::map<unsigned int, std::vector<std::vector<unsigned int>>>();
		transitions[2][2][4] = std::vector<std::vector<unsigned int>>(6);
		transitions[2][2][4][0] = std::vector<unsigned int>(4);
		transitions[2][2][4][0][0] = 4;
		transitions[2][2][4][0][1] = 5;
		transitions[2][2][4][0][2] = 1;
		transitions[2][2][4][0][3] = 0;
		transitions[2][2][4][1] = std::vector<unsigned int>(4);
		transitions[2][2][4][1][0] = 7;
		transitions[2][2][4][1][1] = 5;
		transitions[2][2][4][1][2] = 2;
		transitions[2][2][4][1][3] = 3;
		transitions[2][2][4][2] = std::vector<unsigned int>(4);
		transitions[2][2][4][2][0] = 6;
		transitions[2][2][4][2][1] = 4;
		transitions[2][2][4][2][2] = 5;
		transitions[2][2][4][2][3] = 2;
		transitions[2][2][4][3] = std::vector<unsigned int>(4);
		transitions[2][2][4][3][0] = 6;
		transitions[2][2][4][3][1] = 7;
		transitions[2][2][4][3][2] = 5;
		transitions[2][2][4][3][3] = 2;
		transitions[2][2][4][4] = std::vector<unsigned int>(4);
		transitions[2][2][4][4][0] = 6;
		transitions[2][2][4][4][1] = 7;
		transitions[2][2][4][4][2] = 5;
		transitions[2][2][4][4][3] = 1;
		transitions[2][2][4][5] = std::vector<unsigned int>(4);
		transitions[2][2][4][5][0] = 6;
		transitions[2][2][4][5][1] = 4;
		transitions[2][2][4][5][2] = 5;
		transitions[2][2][4][5][3] = 1;
		transitions[2][1] = std::map<unsigned int, std::vector<std::vector<unsigned int>>>();
		transitions[2][1][3] = std::vector<std::vector<unsigned int>>(3);
		transitions[2][1][3][0] = std::vector<unsigned int>(4);
		transitions[2][1][3][0][0] = 3;
		transitions[2][1][3][0][1] = 1;
		transitions[2][1][3][0][2] = 5;
		transitions[2][1][3][0][3] = 0;
		transitions[2][1][3][1] = std::vector<unsigned int>(4);
		transitions[2][1][3][1][0] = 4;
		transitions[2][1][3][1][1] = 3;
		transitions[2][1][3][1][2] = 2;
		transitions[2][1][3][1][3] = 5;
		transitions[2][1][3][2] = std::vector<unsigned int>(4);
		transitions[2][1][3][2][0] = 4;
		transitions[2][1][3][2][1] = 3;
		transitions[2][1][3][2][2] = 1;
		transitions[2][1][3][2][3] = 5;
		transitions[3] = std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>();
		transitions[3][1] = std::map<unsigned int, std::vector<std::vector<unsigned int>>>();
		transitions[3][1][3] = std::vector<std::vector<unsigned int>>(4);
		transitions[3][1][3][0] = std::vector<unsigned int>(4);
		transitions[3][1][3][0][0] = 4;
		transitions[3][1][3][0][1] = 1;
		transitions[3][1][3][0][2] = 2;
		transitions[3][1][3][0][3] = 0;
		transitions[3][1][3][1] = std::vector<unsigned int>(4);
		transitions[3][1][3][1][0] = 5;
		transitions[3][1][3][1][1] = 4;
		transitions[3][1][3][1][2] = 1;
		transitions[3][1][3][1][3] = 2;
		transitions[3][1][3][2] = std::vector<unsigned int>(4);
		transitions[3][1][3][2][0] = 6;
		transitions[3][1][3][2][1] = 5;
		transitions[3][1][3][2][2] = 4;
		transitions[3][1][3][2][3] = 3;
		transitions[3][1][3][3] = std::vector<unsigned int>(4);
		transitions[3][1][3][3][0] = 6;
		transitions[3][1][3][3][1] = 5;
		transitions[3][1][3][3][2] = 4;
		transitions[3][1][3][3][3] = 2;
	}


	std::vector<Simplex> chooseTriangulation(unsigned int num_dimensions, std::vector<Point>& points, std::vector<unsigned int>& lower_inds, std::vector<unsigned int>& upper_inds, std::vector<unsigned int>& hyper_inds);
	std::vector<Simplex> generateCellSimplices(unsigned int num_dimensions, std::vector<Point>& points);
	std::vector<Point> generateUnitCubePoints(unsigned int num_dimensions);
};

#endif
