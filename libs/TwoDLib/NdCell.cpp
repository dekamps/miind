#include "NdCell.hpp"

NdCell::NdCell(std::vector<unsigned int> _coords, unsigned int _num_dims, std::vector<NdPoint>& _points, Triangulator& _triangulator) :
    grid_coords(_coords),
    num_dimensions(_num_dims),
    triangulator(_triangulator) {
    simplices = generateSimplices(_points);
    hyps = calculateAAHyperplanes(_points);
}

std::vector<Simplex> NdCell::generateSimplices(std::vector<NdPoint>& _points) {
    return triangulator.generateCellSimplices(num_dimensions, _points);
}

double NdCell::getVolume() {
    double vol = 0.0;
    for (Simplex s : simplices)
        vol += s.getVolume();
    return vol;
}

std::map<unsigned int, std::vector<double>> NdCell::calculateAAHyperplanes(std::vector<NdPoint>& _points) {
    std::map<unsigned int, std::vector<double>> out;
    for (unsigned int d = 0; d < num_dimensions; d++) {
        double max = _points[0].coords[d];
        double min = _points[0].coords[d];
        for (unsigned int i = 1; i < _points.size(); i++) {
            if (max < _points[i].coords[d])
                max = _points[i].coords[d];
            if (min > _points[i].coords[d])
                min = _points[i].coords[d];
        }
        std::vector<double> pair(2);
        pair[0] = min;
        pair[1] = max;

        out[d] = pair;
    }
    return out;
}

double NdCell::intersectsWith(NdCell& other) {
    double vol_eps = 0.0000000000001;
    double orig_vol = getVolume();

    std::vector<Simplex> test_simplices = simplices;

    for (auto const& kv : other.hyps) {
        // trivial check if all points are above or below hyperplane
        std::vector<Simplex> new_simplices_1;
        for (Simplex s : test_simplices) {
            if (s.getVolume() < vol_eps)
                continue;

            std::vector<Simplex> st = s.intersectWithHyperplane(kv.first, kv.second[0])[1];

            for (Simplex ns : st)
                new_simplices_1.push_back(ns);
        }
        std::vector<Simplex> new_simplices_2;
        for (Simplex s : new_simplices_1) {
            if (s.getVolume() < vol_eps)
                continue;

            std::vector<Simplex> st = s.intersectWithHyperplane(kv.first, kv.second[1])[0];

            for (Simplex ns : st)
                new_simplices_2.push_back(ns);
        }
        test_simplices = new_simplices_2;
    }

    double vol_prop = 0.0;
    for (Simplex s : test_simplices) {
        vol_prop += s.getVolume();
    }

    if (orig_vol == 0.0)
        return 0.0;

    return vol_prop / orig_vol;
}