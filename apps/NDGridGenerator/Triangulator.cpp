#include "Point.hpp"
#include "Cell.hpp"
#include "Simplex.hpp"
#include "Triangulator.hpp"


std::vector<Simplex> Triangulator::chooseTriangulation(unsigned int num_dimensions, std::vector<Point>& points, std::vector<unsigned int>& lower_inds, std::vector<unsigned int>& upper_inds, std::vector<unsigned int>& hyper_inds) {
	if (lower_inds.size() == 0){
		std::vector<Simplex> out;
		std::vector<Point> ps(upper_inds.size());
		for (unsigned int i=0; i<upper_inds.size(); i++)
			ps[i] = points[upper_inds[i]];
		out.push_back(Simplex(num_dimensions, ps, *this));
		return out;
	}
	if (upper_inds.size() == 0){
		std::vector<Simplex> out;
		std::vector<Point> ps(lower_inds.size());
		for (unsigned int i=0; i<lower_inds.size(); i++)
			ps[i] = points[lower_inds[i]];
		out.push_back(Simplex(num_dimensions, ps, *this));
		return out;
	}
	std::vector<std::vector<unsigned int>> tris = transitions[lower_inds.size()][upper_inds.size()][hyper_inds.size()];
	std::vector<Simplex> out;
	for (unsigned int t=0; t <tris.size(); t++) {
		std::vector<Point> ps(tris[t].size());
		for (unsigned int i=0; i<tris[t].size(); i++)
			ps[i] = points[tris[t][i]];
		out.push_back(Simplex(num_dimensions, ps, *this));
	}
	return out;
}

std::vector<Simplex> Triangulator::generateCellSimplices(unsigned int num_dimensions, std::vector<Point>& points) {
	switch(num_dimensions) {
	case 2: {
		std::vector<Simplex> simplices;
		std::vector<Point> ps_0(3);
		ps_0[0] = points[0];
		ps_0[1] = points[1];
		ps_0[2] = points[3];
		simplices.push_back(Simplex(num_dimensions,ps_0,*this));
		std::vector<Point> ps_1(3);
		ps_1[0] = points[3];
		ps_1[1] = points[2];
		ps_1[2] = points[0];
		simplices.push_back(Simplex(num_dimensions,ps_1,*this));
		return simplices;
	}
	case 3: {
		std::vector<Simplex> simplices;
		std::vector<Point> ps_0(4);
		ps_0[0] = points[0];
		ps_0[1] = points[1];
		ps_0[2] = points[3];
		ps_0[3] = points[7];
		simplices.push_back(Simplex(num_dimensions,ps_0,*this));
		std::vector<Point> ps_1(4);
		ps_1[0] = points[7];
		ps_1[1] = points[1];
		ps_1[2] = points[5];
		ps_1[3] = points[0];
		simplices.push_back(Simplex(num_dimensions,ps_1,*this));
		std::vector<Point> ps_2(4);
		ps_2[0] = points[7];
		ps_2[1] = points[2];
		ps_2[2] = points[3];
		ps_2[3] = points[0];
		simplices.push_back(Simplex(num_dimensions,ps_2,*this));
		std::vector<Point> ps_3(4);
		ps_3[0] = points[0];
		ps_3[1] = points[2];
		ps_3[2] = points[6];
		ps_3[3] = points[7];
		simplices.push_back(Simplex(num_dimensions,ps_3,*this));
		std::vector<Point> ps_4(4);
		ps_4[0] = points[0];
		ps_4[1] = points[4];
		ps_4[2] = points[5];
		ps_4[3] = points[7];
		simplices.push_back(Simplex(num_dimensions,ps_4,*this));
		std::vector<Point> ps_5(4);
		ps_5[0] = points[7];
		ps_5[1] = points[4];
		ps_5[2] = points[6];
		ps_5[3] = points[0];
		simplices.push_back(Simplex(num_dimensions,ps_5,*this));
		return simplices;
	}
	default: {
		return std::vector<Simplex>();
	}
	}
}
std::vector<Point> Triangulator::generateUnitCubePoints(unsigned int num_dimensions) {
	switch(num_dimensions) {
	case 2: {
		std::vector<Point> points(4);
		std::vector<double> coords_0(2);
		coords_0[0] = 0.0;
		coords_0[1] = 0.0;
		points[0] = Point(coords_0);
		std::vector<double> coords_1(2);
		coords_1[0] = 0.0;
		coords_1[1] = 1.0;
		points[1] = Point(coords_1);
		std::vector<double> coords_2(2);
		coords_2[0] = 1.0;
		coords_2[1] = 0.0;
		points[2] = Point(coords_2);
		std::vector<double> coords_3(2);
		coords_3[0] = 1.0;
		coords_3[1] = 1.0;
		points[3] = Point(coords_3);
		return points;
	}
	case 3: {
		std::vector<Point> points(8);
		std::vector<double> coords_0(3);
		coords_0[0] = 0.0;
		coords_0[1] = 0.0;
		coords_0[2] = 0.0;
		points[0] = Point(coords_0);
		std::vector<double> coords_1(3);
		coords_1[0] = 0.0;
		coords_1[1] = 0.0;
		coords_1[2] = 1.0;
		points[1] = Point(coords_1);
		std::vector<double> coords_2(3);
		coords_2[0] = 0.0;
		coords_2[1] = 1.0;
		coords_2[2] = 0.0;
		points[2] = Point(coords_2);
		std::vector<double> coords_3(3);
		coords_3[0] = 0.0;
		coords_3[1] = 1.0;
		coords_3[2] = 1.0;
		points[3] = Point(coords_3);
		std::vector<double> coords_4(3);
		coords_4[0] = 1.0;
		coords_4[1] = 0.0;
		coords_4[2] = 0.0;
		points[4] = Point(coords_4);
		std::vector<double> coords_5(3);
		coords_5[0] = 1.0;
		coords_5[1] = 0.0;
		coords_5[2] = 1.0;
		points[5] = Point(coords_5);
		std::vector<double> coords_6(3);
		coords_6[0] = 1.0;
		coords_6[1] = 1.0;       
		coords_6[2] = 0.0; 
		points[6] = Point(coords_6);
		std::vector<double> coords_7(3);
		coords_7[0] = 1.0;
		coords_7[1] = 1.0;
		coords_7[2] = 1.0;
		points[7] = Point(coords_7);
		return points;
	}
	default: {
		return std::vector<Point>();
	}
	}
}
