#ifndef MIINDGEN_CELL_HPP
#define MIINDGEN_CELL_HPP

#include "Simplex.hpp"
#include <cmath>

class NdCell {
public:
    std::vector<unsigned int> grid_coords;
    unsigned int num_dimensions;
    Triangulator& triangulator;
    std::vector<Simplex> simplices;
    std::map<unsigned int, std::vector<double>> hyps;

    NdCell(std::vector<unsigned int> _coords, unsigned int _num_dims, std::vector<NdPoint>& _points, Triangulator& _triangulator);
    std::vector<Simplex> generateSimplices(std::vector<NdPoint>& _points);
    double getVolume();
    std::map<unsigned int, std::vector<double>> calculateAAHyperplanes(std::vector<NdPoint>& _points);
    double intersectsWith(NdCell& other);
};

#endif