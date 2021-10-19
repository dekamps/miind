#ifndef APP_ND_GRID_SIMPLEX
#define APP_ND_GRID_SIMPLEX

#include "NdPoint.hpp"
#include "Triangulator.hpp"

#include <stdlib.h>
#include <boost/numeric/ublas/matrix.hpp> 
#include <boost/numeric/ublas/io.hpp> 
#include <boost/numeric/ublas/matrix_proxy.hpp> 
#include <boost/numeric/ublas/lu.hpp> 

class Simplex {
public:
    Triangulator& triangulator;
    unsigned int num_dimensions;
    std::vector<NdPoint> points;
    std::vector<NdPoint> lines;

    Simplex(unsigned int num_dims, std::vector<std::vector<double>>& _points, Triangulator& _triangulator);
    Simplex(unsigned int num_dims, std::vector<NdPoint> _points, Triangulator& _triangulator);
    Simplex(const Simplex& other);
    Simplex& operator=(const Simplex& other);
    std::vector<NdPoint> generateLines();

    // CalcDeterminant by Richel Bilderbeek : http://www.richelbilderbeek.nl/CppUblasMatrixExample7.htm
    double CalcDeterminant(boost::numeric::ublas::matrix<double> m);
    double getVolume();
    std::vector<std::vector<Simplex>> intersectWithHyperplane(unsigned int dim_index, double dim);
};

#endif