#include "NdGrid.hpp"

NdGrid::NdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
    double _threshold_v, double _reset_v, double _timestep) :
    base(_base),
    dimensions(_dims),
    resolution(_res),
    threshold_v(_threshold_v),
    reset_v(_reset_v),
    timestep(_timestep) {

    num_dimensions = _dims.size();

    generate_cell_coords(std::vector<unsigned int>(), resolution);
}

void NdGrid::setCppFunction(void(*Func)(NdPoint&, double)) {
    fcnPtr = Func;
}

NdCell NdGrid::generate_cell_with_coords(std::vector<unsigned int> cell_coord, bool btranslated) {

    std::vector<double> base_point_coords(num_dimensions);
    for (unsigned int j = 0; j < num_dimensions; j++) {
        base_point_coords[j] = base[j] + (cell_coord[j] * (dimensions[j] / resolution[j]));
    }

    std::vector<NdPoint> ps = triangulator.generateUnitCubePoints(num_dimensions);
    for (unsigned int i = 0; i < ps.size(); i++) {

        for (unsigned int d = 0; d < num_dimensions; d++) {
            ps[i].coords[d] *= (dimensions[d] / resolution[d]);
            ps[i].coords[d] += base_point_coords[d];
        }

        if (btranslated)
            fcnPtr(ps[i], timestep);
    }

    return NdCell(cell_coord, num_dimensions, ps, triangulator);
}

void NdGrid::generate_cell_coords(std::vector<unsigned int> cell_coord, std::vector<unsigned int> res) {
    unsigned int res_head = res[0];

    if (res.size() < 1) // We don't work with 1D
        return;

    std::vector<unsigned int> res_tail(res.size() - 1);

    for (unsigned int i = 0; i < res_tail.size(); i++)
        res_tail[i] = res[i + 1];

    if (res.size() == 1) {
        for (unsigned int d = 0; d < res_head; d++) {
            std::vector<unsigned int> full_coord(cell_coord.size() + 1);
            for (unsigned int c = 0; c < cell_coord.size(); c++)
                full_coord[c] = cell_coord[c];
            full_coord[num_dimensions - 1] = d;

            coord_list.push_back(full_coord);
        }

        return;
    }

    for (unsigned int d = 0; d < res_head; d++) {
        std::vector<unsigned int> full_coord(cell_coord.size() + 1);
        for (unsigned int c = 0; c < cell_coord.size(); c++)
            full_coord[c] = cell_coord[c];
        full_coord[full_coord.size() - 1] = d;
        generate_cell_coords(full_coord, res_tail);
    }
}

std::vector<unsigned int> NdGrid::coords_to_strip_and_cell(std::vector<unsigned int> coords) {
    unsigned int index = 0;
    unsigned int multiplicand = 1;

    // MIIND expects membrane potential or the variable which receives instantaneous synaptic potentials
    // to be the last coordinate. So we're working back to front here.
    std::vector<unsigned int> coords_rev(num_dimensions);
    std::vector<unsigned int> res_rev(num_dimensions);
    for (unsigned int i = 0; i < num_dimensions; i++) {
        // coords_rev[num_dimensions-1-i] = coords[i];
        // res_rev[num_dimensions-1-i] = resolution[i];
        coords_rev[i] = coords[i];
        res_rev[i] = resolution[i];
    }

    for (unsigned int res : res_rev)
        multiplicand *= res;
    for (int d = 0; d < num_dimensions - 1; d++) {
        multiplicand /= res_rev[d];
        index += int(coords_rev[d] * multiplicand);
    }
    index /= res_rev[num_dimensions - 1];
    std::vector<unsigned int> pair(2);
    pair[0] = index;
    pair[1] = coords_rev[num_dimensions - 1];
    return pair;
}

unsigned int NdGrid::coords_to_index(std::vector<unsigned int> coords) {
    unsigned int index = 0;
    unsigned int operand = 1;
    for (int c = num_dimensions - 1; c >= 0; c--) {
        index += coords[c] * operand;
        operand *= resolution[c];
    }
    return index;
}

std::vector<unsigned int> NdGrid::index_to_coords(unsigned int index) {
    std::vector<unsigned int> coords(num_dimensions);
    unsigned int operand = 1;

    std::vector<unsigned int> operands(num_dimensions);

    for (int c = num_dimensions - 1; c >= 0; c--) {
        operands[c] = operand;
        operand *= resolution[c];
    }

    for (int c = 0; c < num_dimensions - 1; c++) {
        unsigned int m = int(index / operands[c]);
        coords[c] = m;
        index = index % operands[c];
    }
    coords[num_dimensions - 1] = index;
    return coords;
}

std::vector<unsigned int> NdGrid::getCellCoordsForPoint(NdPoint& p) {
    std::vector<unsigned int> coords(num_dimensions);
    for (unsigned int c = 0; c < num_dimensions; c++) {
        int co = int((p.coords[c] - base[c]) / (dimensions[c] / resolution[c]));
        if (co >= int(resolution[c]))
            co = resolution[c] - 1;
        if (co < 0)
            co = 0;
        coords[c] = co;
    }
    return coords;
}

void NdGrid::buildCellRange(std::vector<NdCell>& cell_ptrs, std::vector<unsigned int> base_min, std::vector<unsigned int> max_coords, std::vector<unsigned int> min_coords) {

    if (max_coords.size() == 1) {
        for (unsigned int i = 0; i < (max_coords[0] - min_coords[0]) + 1; i++) {
            std::vector<unsigned int> nb = base_min;
            nb.push_back(min_coords[0] + i);
            cell_ptrs.push_back(generate_cell_with_coords(nb, false));
        }
    }
    else  if (max_coords.size() > 1) {
        for (unsigned int i = 0; i < (max_coords[0] - min_coords[0]) + 1; i++) {
            std::vector<unsigned int> nb = base_min;
            nb.push_back(min_coords[0] + i);

            std::vector<unsigned int> max_tail(max_coords.size() - 1);
            for (unsigned int j = 0; j < max_coords.size() - 1; j++)
                max_tail[j] = max_coords[j + 1];

            std::vector<unsigned int> min_tail(min_coords.size() - 1);
            for (unsigned int j = 0; j < min_coords.size() - 1; j++)
                min_tail[j] = min_coords[j + 1];

            buildCellRange(cell_ptrs, nb, max_tail, min_tail);
        }
    }
}

std::vector<NdCell> NdGrid::getCellRange(NdCell& tcell) {

    std::vector<double> max = tcell.simplices[0].points[0].coords;
    std::vector<double> min = tcell.simplices[0].points[0].coords;

    for (unsigned int c = 0; c < num_dimensions; c++) {
        for (Simplex s : tcell.simplices) {
            for (NdPoint p : s.points) {
                if (max[c] < p.coords[c])
                    max[c] = p.coords[c];
                if (min[c] > p.coords[c])
                    min[c] = p.coords[c];
            }
        }
    }

    NdPoint p_max = NdPoint(max);
    NdPoint p_min = NdPoint(min);

    std::vector<unsigned int> max_coords = getCellCoordsForPoint(p_max);
    std::vector<unsigned int> min_coords = getCellCoordsForPoint(p_min);

    std::vector<NdCell> cells;
    buildCellRange(cells, std::vector<unsigned int>(), max_coords, min_coords);
    return cells;
}

std::map<std::vector<unsigned int>, double>
NdGrid::calculateTransitionForCell(NdCell& tcell, std::vector<NdCell>& cell_range) {
    std::map<std::vector<unsigned int>, double> t;
    unsigned int threshold_cell = int((threshold_v - base[num_dimensions - 1]) / (dimensions[num_dimensions - 1] / resolution[num_dimensions - 1]));
    for (NdCell check_cell : cell_range) {
        double prop = tcell.intersectsWith(check_cell);
        if (prop == 0)
            continue;
        if (check_cell.grid_coords[num_dimensions - 1] > threshold_cell)
            check_cell.grid_coords[num_dimensions - 1] = threshold_cell;
        if (t.find(check_cell.grid_coords) == t.end())
            t[check_cell.grid_coords] = 0.0;

        t[check_cell.grid_coords] += prop;
    }
    return t;
}

std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> NdGrid::calculateTransitionMatrix() {
    std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> transitions;
#pragma omp parallel for
    for (int c = 0; c < coord_list.size(); c++) {
        NdCell cell = generate_cell_with_coords(coord_list[c], true);
        std::vector<NdCell> check_cells = getCellRange(cell);
        std::map<std::vector<unsigned int>, double> ts = calculateTransitionForCell(cell, check_cells);

        if (ts.size() == 0) { // cell was completely outside the grid, so don't move it.
            ts[cell.grid_coords] = 1.0;
        }

        double total_prop = 0.0;
        for (auto const& kv : ts) {
            total_prop += kv.second;
        }
        double missed_prop = 1.0 / total_prop;

        for (auto kv : ts) {
            double d = ts[kv.first];
            ts[kv.first] *= missed_prop;
        }
        transitions[cell.grid_coords] = ts;
        if (transitions.size() % 100 == 0)
            std::cout << transitions.size() << " complete(ish).\n";
    }

    return transitions;
}

void NdGrid::generateTMatFileBatched(std::string basename) {
    unsigned int batch_size = 1000;
    std::ofstream file;
    file.open(basename + ".tmat");

    file << "0\t0\n";

    std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> transitions;
    for (unsigned int batch = 0; batch < (coord_list.size() / batch_size) + 1; batch++) {
#pragma omp parallel for
        for (int c = (batch * batch_size); c < (batch * batch_size) + batch_size; c++) {
            if (c >= coord_list.size())
                continue;

            NdCell cell = generate_cell_with_coords(coord_list[c], true);
            std::vector<NdCell> check_cells = getCellRange(cell);
            std::map<std::vector<unsigned int>, double> ts = calculateTransitionForCell(cell, check_cells);

            if (ts.size() == 0) { // cell was completely outside the grid, so don't move it.
                ts[cell.grid_coords] = 1.0;
            }

            double total_prop = 0.0;
            for (auto const& kv : ts) {
                total_prop += kv.second;
            }
            double missed_prop = 1.0 / total_prop;

            for (auto const& kv : ts) {
                double d = ts[kv.first];
                ts[kv.first] *= missed_prop;
            }
#pragma omp critical
            transitions[cell.grid_coords] = ts;
        }
        std::cout << '\r' << std::setw(5) << 100.0 * ((float)(batch * batch_size) / (float)coord_list.size()) << "% complete." << std::setfill(' ') << std::flush;
    }

    for (auto const& kv : transitions) {
        std::vector<unsigned int> pair = coords_to_strip_and_cell(kv.first);
        file << "1000000000;" << pair[0] << "," << pair[1] << ";";
        for (auto const& tv : kv.second) {
            std::vector<unsigned int> tpair = coords_to_strip_and_cell(tv.first);
            file << tpair[0] << "," << tpair[1] << ":" << tv.second << ";";
        }
        file << "\n";
    }

    transitions.clear();

    std::cout << "\n";
}

void NdGrid::generateTMatFileLowMemory(std::string basename) {
    std::ofstream file;
    file.open(basename + ".tmat");

    file << "0\t0\n";
    unsigned int num_lines = 0;
    for (std::vector<unsigned int> c : coord_list) {
        NdCell cell = generate_cell_with_coords(c, true);
        std::vector<unsigned int> pair = coords_to_strip_and_cell(cell.grid_coords);
        file << "1000000000;" << pair[0] << "," << pair[1] << ";";
        std::vector<NdCell> check_cells = getCellRange(cell);
        std::map<std::vector<unsigned int>, double> ts = calculateTransitionForCell(cell, check_cells);

        if (ts.size() == 0) { // cell was completely outside the grid, so don't move it.
            ts[cell.grid_coords] = 1.0;
        }

        double total_prop = 0.0;
        for (auto const& kv : ts) {
            total_prop += kv.second;
        }
        double missed_prop = 1.0 / total_prop;

        for (auto const& kv : ts) {
            double d = ts[kv.first];
            ts[kv.first] *= missed_prop;
            std::vector<unsigned int> tpair = coords_to_strip_and_cell(kv.first);
            file << tpair[0] << "," << tpair[1] << ":" << kv.second << ";";
        }

        file << "\n";
        num_lines++;
        if (num_lines % 100 == 0)
            std::cout << num_lines << " complete.\n";
    }

    file.close();
}

void NdGrid::generateTMatFile(std::string basename) {

    std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> trs = calculateTransitionMatrix();

    std::ofstream file;
    file.open(basename + ".tmat");

    file << "0\t0\n";
    for (auto const& kv : trs) {
        std::vector<unsigned int> pair = coords_to_strip_and_cell(kv.first);
        file << "1000000000;" << pair[0] << "," << pair[1] << ";";
        for (auto const& tv : kv.second) {
            std::vector<unsigned int> tpair = coords_to_strip_and_cell(tv.first);
            file << tpair[0] << "," << tpair[1] << ":" << tv.second << ";";
        }
        file << "\n";
    }

    file.close();
}

void NdGrid::generateResetMapping(std::ofstream& file) {
    unsigned int num_strips = 1;
    for (unsigned int d = 0; d < num_dimensions - 1; d++)
        num_strips *= resolution[d];

    unsigned int threshold_cell = int((threshold_v - base[num_dimensions - 1]) / (dimensions[num_dimensions - 1] / resolution[num_dimensions - 1]));
    unsigned int reset_cell = int((reset_v - base[num_dimensions - 1]) / (dimensions[num_dimensions - 1] / resolution[num_dimensions - 1]));

    for (unsigned int strip = 0; strip < num_strips; strip++) {
        file << strip << "," << threshold_cell << "\t" << strip << "," << reset_cell << "\t" << 1 << "\n";
    }
}

void NdGrid::generateModelFile(std::string basename, double timestep_multiplier) {
    std::ofstream file;
    file.open(basename + ".model");

    file << "<Model>\n";
    file << "<Mesh>\n";
    file << "<TimeStep>" << timestep * timestep_multiplier << "</TimeStep>\n";

    file << "<GridNumDimensions>" << num_dimensions << "</GridNumDimensions>\n";
    file << "<GridDimensions>";
    for (unsigned int n = 0; n < num_dimensions; n++) {
        file << dimensions[n] << " ";
    }
    file << "</GridDimensions>\n";

    file << "<GridResolution>";
    for (unsigned int n = 0; n < num_dimensions; n++) {
        file << resolution[n] << " ";
    }
    file << "</GridResolution>\n";

    file << "<GridBase>";
    for (unsigned int n = 0; n < num_dimensions; n++) {
        file << base[n] << " ";
    }
    file << "</GridBase>\n";

    file << "</Mesh>\n";
    file << "<Stationary>\n";
    file << "</Stationary>\n";
    file << "<Mapping type = \"Reversal\">\n";
    file << "</Mapping>\n";
    file << "<threshold>" << threshold_v << "</threshold>\n";
    file << "<V_reset>" << reset_v << "</V_reset>\n";

    file << "<Mapping type=\"Reset\">\n";
    generateResetMapping(file);
    file << "</Mapping>\n";
    file << "</Model>\n";

    file.close();
}