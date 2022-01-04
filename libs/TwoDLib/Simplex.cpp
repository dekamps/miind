#include "Simplex.hpp"

Simplex::Simplex(unsigned int num_dims, std::vector<std::vector<double>>& _points, Triangulator& _triangulator) :
    num_dimensions(num_dims),
    points(_points.size()),
    lines(0),
    triangulator(_triangulator) {
    for (unsigned int i = 0; i < _points.size(); i++) {
        points[i] = NdPoint(_points[i]);
    }

    lines = generateLines();
}

Simplex::Simplex(unsigned int num_dims, std::vector<NdPoint> _points, Triangulator& _triangulator) :
    num_dimensions(num_dims),
    points(_points),
    lines(0),
    triangulator(_triangulator) {

    lines = generateLines();
}

Simplex::Simplex(const Simplex& other) :
    num_dimensions(other.num_dimensions),
    triangulator(other.triangulator) {
    points = std::vector<NdPoint>(other.points.size());
    for (unsigned int i = 0; i < other.points.size(); i++) {
        points[i] = other.points[i];
    }

    lines = std::vector<NdPoint>(other.lines.size());
    for (unsigned int i = 0; i < other.lines.size(); i++) {
        lines[i] = other.lines[i];
    }
}

Simplex& Simplex::operator=(const Simplex& other) {
    num_dimensions = other.num_dimensions;
    triangulator = other.triangulator;
    points = std::vector<NdPoint>(other.points.size());
    for (unsigned int i = 0; i < other.points.size(); i++) {
        points[i] = other.points[i];
    }

    lines = std::vector<NdPoint>(other.lines.size());
    for (unsigned int i = 0; i < other.lines.size(); i++) {
        lines[i] = other.lines[i];
    }

    return *this;
}

std::vector<NdPoint> Simplex::generateLines() {
    std::vector<NdPoint> lines(num_dimensions);
    for (unsigned int p = 0; p < points.size() - 1; p++) {
        std::vector<double> coords(num_dimensions);
        for (unsigned int c = 0; c < num_dimensions; c++)
            coords[c] = points[p + 1].coords[c] - points[0].coords[c];
        lines[p] = NdPoint(coords);
    }
    return lines;
}

// CalcDeterminant by Richel Bilderbeek : http://www.richelbilderbeek.nl/CppUblasMatrixExample7.htm
double Simplex::CalcDeterminant(boost::numeric::ublas::matrix<double> m)
{
    assert(m.size1() == m.size2() && "Can only calculate the determinant of square matrices");
    boost::numeric::ublas::permutation_matrix<std::size_t> pivots(m.size1());

    const int is_singular = boost::numeric::ublas::lu_factorize(m, pivots);

    if (is_singular) return 0.0;

    double d = 1.0;
    const std::size_t sz = pivots.size();
    for (std::size_t i = 0; i != sz; ++i)
    {
        if (pivots(i) != i)
        {
            d *= -1.0;
        }
        d *= m(i, i);
    }
    return d;
}

double Simplex::getVolume() {
    boost::numeric::ublas::matrix<double> m(num_dimensions, num_dimensions);
    for (unsigned int l = 0; l < num_dimensions; l++) {
        for (unsigned int c = 0; c < num_dimensions; c++) {
            m(l, c) = lines[l].coords[c];
        }
    }

    unsigned int dim_fac = 0;
    for (unsigned int n = 0; n < num_dimensions; n++)
        dim_fac += n;

    return std::abs(CalcDeterminant(m) / dim_fac) / 2;
}

std::vector<std::vector<Simplex>> Simplex::intersectWithHyperplane(unsigned int dim_index, double dim) {
    double eps = 0.000000000001;

    std::vector<NdPoint*> lower;
    std::vector<NdPoint*> upper;
    std::vector<NdPoint*> equal;
    for (unsigned int i = 0; i < points.size(); i++) {
        if (points[i].coords[dim_index] < dim - eps) lower.push_back(&points[i]);
        else if (points[i].coords[dim_index] > dim + eps) upper.push_back(&points[i]);
        else equal.push_back(&points[i]);
    }

    std::vector<NdPoint> p_outs;
    for (NdPoint* p0 : lower) {
        for (NdPoint* p1 : upper) {
            double t = (dim - p0->coords[dim_index]) / (p1->coords[dim_index] - p0->coords[dim_index]);
            std::vector<double> coords(num_dimensions);
            for (unsigned int i = 0; i < num_dimensions; i++) {
                coords[i] = p0->coords[i] + ((p1->coords[i] - p0->coords[i]) * t);
            }
            NdPoint np(coords);
            np.hyper = true;
            p_outs.push_back(np);
        }
    }

    if (p_outs.size() == 0) {
        std::vector<std::vector<Simplex>> out;
        bool points_above = true;
        for (NdPoint p : points)
            points_above &= p.coords[dim_index] >= dim - eps;

        std::vector<Simplex> less;
        std::vector<Simplex> greater;

        if (!points_above) {
            less.push_back(Simplex(num_dimensions, points, triangulator));
            out.push_back(less);
            out.push_back(std::vector<Simplex>());
        }
        else {
            greater.push_back(Simplex(num_dimensions, points, triangulator));
            out.push_back(std::vector<Simplex>());
            out.push_back(greater);
        }
        return out;
    }

    unsigned int index = 0;
    std::vector<unsigned int> i_less(lower.size());
    for (unsigned int i = 0; i < lower.size(); i++) {
        i_less[i] = i + index;
    }
    index += lower.size();

    std::vector<unsigned int> i_greater(upper.size());
    for (unsigned int i = 0; i < upper.size(); i++) {
        i_greater[i] = i + index;
    }
    index += upper.size();

    std::vector<unsigned int> i_hyp(p_outs.size());
    for (unsigned int i = 0; i < p_outs.size(); i++) {
        i_hyp[i] = i + index;
    }
    index += p_outs.size();

    for (unsigned int i = 0; i < equal.size(); i++) i_hyp.push_back(i + index);

    std::vector<NdPoint> p_total(lower.size() + upper.size() + p_outs.size() + equal.size());
    for (unsigned int i = 0; i < lower.size(); i++) {
        p_total[i] = *(lower[i]);
    }
    for (unsigned int i = 0; i < upper.size(); i++) {
        p_total[lower.size() + i] = *(upper[i]);
    }
    for (unsigned int i = 0; i < p_outs.size(); i++) {
        p_total[lower.size() + upper.size() + i] = p_outs[i];
    }
    for (unsigned int i = 0; i < equal.size(); i++) {
        p_total[lower.size() + upper.size() + p_outs.size() + i] = *(equal[i]);
    }

    std::vector<Simplex> simplices = triangulator.chooseTriangulation(num_dimensions, p_total, i_less, i_greater, i_hyp);

    std::vector<Simplex> less;
    std::vector<Simplex> greater;
    for (Simplex s : simplices) {
        bool all_above = true;
        bool all_below = true;
        for (NdPoint p : s.points) {
            all_above &= p.coords[dim_index] >= dim - eps;
            all_below &= p.coords[dim_index] <= dim + eps;
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