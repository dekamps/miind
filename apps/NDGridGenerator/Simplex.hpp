#ifndef APP_ND_GRID_SIMPLEX
#define APP_ND_GRID_SIMPLEX

#include "Point.hpp"
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
    std::vector<Point> points;
    std::vector<Point> lines;

    Simplex(unsigned int num_dims, std::vector<std::vector<double>>& _points, Triangulator& _triangulator):
    num_dimensions(num_dims),
    points(_points.size()),
    lines(0),
    triangulator(_triangulator) {
        for (unsigned int i=0; i<_points.size(); i++) {
            points[i] = Point(_points[i]);
        }

        lines = generateLines();
    }

    Simplex(unsigned int num_dims, std::vector<Point> _points, Triangulator& _triangulator):
    num_dimensions(num_dims),
    points(_points),
    lines(0),
    triangulator(_triangulator) {

        lines = generateLines();
    }

    Simplex(const Simplex& other) :
    num_dimensions(other.num_dimensions),
    triangulator(other.triangulator) {
        points = std::vector<Point>(other.points.size());
        for(unsigned int i=0; i<other.points.size(); i++) {
            points[i] = other.points[i];
        }

        lines = std::vector<Point>(other.lines.size());
        for(unsigned int i=0; i<other.lines.size(); i++) {
            lines[i] = other.lines[i];
        }
    }

    Simplex& operator=(const Simplex &other) {
        num_dimensions = other.num_dimensions;
        triangulator = other.triangulator;
        points = std::vector<Point>(other.points.size());
        for(unsigned int i=0; i<other.points.size(); i++) {
            points[i] = other.points[i];
        }

        lines = std::vector<Point>(other.lines.size());
        for(unsigned int i=0; i<other.lines.size(); i++) {
            lines[i] = other.lines[i];
        }

        return *this;
    }

    std::vector<Point> generateLines() {
        std::vector<Point> lines(num_dimensions);
        for(unsigned int p=0; p<points.size()-1; p++) {
            std::vector<double> coords(num_dimensions);
            for (unsigned int c=0; c<num_dimensions; c++)
                coords[c] = points[p+1].coords[c] - points[0].coords[c];
            lines[p] = Point(coords);
        }
        return lines;
    }

    // CalcDeterminant by Richel Bilderbeek : http://www.richelbilderbeek.nl/CppUblasMatrixExample7.htm
    double CalcDeterminant(boost::numeric::ublas::matrix<double> m) 
    { 
        assert(m.size1() == m.size2() && "Can only calculate the determinant of square matrices"); 
        boost::numeric::ublas::permutation_matrix<std::size_t> pivots(m.size1() ); 

        const int is_singular = boost::numeric::ublas::lu_factorize(m, pivots); 

        if (is_singular) return 0.0; 

        double d = 1.0; 
        const std::size_t sz = pivots.size(); 
        for (std::size_t i=0; i != sz; ++i) 
        { 
            if (pivots(i) != i) 
            { 
            d *= -1.0; 
            } 
            d *= m(i,i); 
        } 
        return d; 
    } 

    double getVolume() {
        boost::numeric::ublas::matrix<double> m(num_dimensions,num_dimensions);
        for (unsigned int l=0; l<num_dimensions; l++) {
            for(unsigned int c=0; c<num_dimensions; c++) {
                m(l,c) = lines[l].coords[c];
            }
        }

        unsigned int dim_fac = 0;
        for(unsigned int n=0; n<num_dimensions; n++)
            dim_fac += n;

        return std::abs(CalcDeterminant(m)/dim_fac)/2;
    }

    std::vector<std::vector<Simplex>> intersectWithHyperplane(unsigned int dim_index, double dim) {
        double eps = 0.00000000001;

        std::vector<Point*> lower;
        std::vector<Point*> upper;
        std::vector<Point*> equal;
        for (unsigned int i=0; i<points.size(); i++) {
            if(points[i].coords[dim_index] < dim - eps) lower.push_back(&points[i]);
            else if(points[i].coords[dim_index] > dim + eps) upper.push_back(&points[i]);
            else equal.push_back(&points[i]);
        }

        std::vector<Point> p_outs;
        for (Point* p0 : lower){
            for (Point* p1 : upper) {
                double t = (dim - p0->coords[dim_index]) / (p1->coords[dim_index] - p0->coords[dim_index]);
                std::vector<double> coords(num_dimensions);
                for (unsigned int i=0; i<num_dimensions; i++){
                    coords[i] = p0->coords[i] + ((p1->coords[i] - p0->coords[i])*t);
                }
                Point np(coords);
                np.hyper = true;
                p_outs.push_back(np);
            }
        }

        if (p_outs.size() == 0) {
            std::vector<std::vector<Simplex>> out;
            bool points_above = true;
            for(Point p : points) 
                points_above &= p.coords[dim_index] >= dim - eps;

            std::vector<Simplex> less;
            std::vector<Simplex> greater;

            if (!points_above){
                less.push_back(Simplex(num_dimensions, points, triangulator));
                out.push_back(less);
                out.push_back(std::vector<Simplex>());
            } else {
                greater.push_back(Simplex(num_dimensions, points, triangulator));
                out.push_back(std::vector<Simplex>());
                out.push_back(greater); 
            }     
            return out;
        }

        unsigned int index = 0;
        std::vector<unsigned int> i_less(lower.size());
        for (unsigned int i=0; i<lower.size(); i++) {
            i_less[i] = i + index;
        }
        index += lower.size();

        std::vector<unsigned int> i_greater(upper.size());
        for (unsigned int i=0; i<upper.size(); i++) {
            i_greater[i] = i + index;
        } 
        index += upper.size();

        std::vector<unsigned int> i_hyp(p_outs.size());
        for (unsigned int i=0; i<p_outs.size(); i++) {
            i_hyp[i] = i + index;
        } 
        index += p_outs.size();
        
        for (unsigned int i=0; i<equal.size(); i++) i_hyp.push_back(i + index);

        std::vector<Point> p_total(lower.size()+upper.size()+p_outs.size()+equal.size());
        for(unsigned int i=0; i<lower.size(); i++){
            p_total[i] = *(lower[i]);
        }
        for(unsigned int i=0; i<upper.size(); i++){
            p_total[lower.size()+i] = *(upper[i]);
        }
        for(unsigned int i=0; i<p_outs.size(); i++){
            p_total[lower.size()+upper.size()+i] = p_outs[i];
        }
        for(unsigned int i=0; i<equal.size(); i++){
            p_total[lower.size()+upper.size()+p_outs.size()+i] = *(equal[i]);
        }

        std::vector<Simplex> simplices = triangulator.chooseTriangulation(num_dimensions, p_total, i_less, i_greater, i_hyp);

        std::vector<Simplex> less;
        std::vector<Simplex> greater;
        for (Simplex s : simplices){
            bool all_above = true;
            bool all_below = true;
            for (Point p : s.points) {
                all_above &= p.coords[dim_index] >= dim-eps;
                all_below &= p.coords[dim_index] <= dim+eps;
            }

            if (all_above)
                greater.push_back(s);

            if (all_below)
                less.push_back(s);
        }

        std::vector<std::vector<Simplex>> out;
        out.push_back(less);
        out.push_back(greater);
        return out;
    }
};

#endif