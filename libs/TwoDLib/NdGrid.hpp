#ifndef MIINDGEN_GRID_HPP
#define MIINDGEN_GRID_HPP

#ifdef ENABLE_OMP
#include <omp.h>
#endif

#include "NdPoint.hpp"
#include "Simplex.hpp"
#include "NdCell.hpp"
#include "Triangulator.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

class NdGrid {
public:
    double timestep;
    unsigned int num_dimensions;
    double threshold_v;
    double reset_v;
    Triangulator triangulator;
    std::vector<double> dimensions;
    std::vector<unsigned int> resolution;
    std::vector<double> base;
    std::vector<std::vector<unsigned int>> coord_list;

    void (*fcnPtr)(NdPoint&,double);

    NdGrid(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
        double _threshold_v, double _reset_v, double _timestep);
    void setCppFunction(void(*Func)(NdPoint&, double));
    virtual NdCell generate_cell_with_coords(std::vector<unsigned int> cell_coord, bool btranslated);
    void generate_cell_coords(std::vector<unsigned int> cell_coord, std::vector<unsigned int> res);
    std::vector<unsigned int> coords_to_strip_and_cell(std::vector<unsigned int> coords);
    unsigned int coords_to_index(std::vector<unsigned int> coords);
    std::vector<unsigned int> index_to_coords(unsigned int index);
    std::vector<unsigned int> getCellCoordsForPoint(NdPoint& p);
    void buildCellRange(std::vector<NdCell>& cell_ptrs, std::vector<unsigned int> base_min,
        std::vector<unsigned int> max_coords, std::vector<unsigned int> min_coords);
    std::vector<NdCell> getCellRange(NdCell& tcell);
    std::map<std::vector<unsigned int>, double> calculateTransitionForCell(NdCell& tcell, std::vector<NdCell>& cell_range);
    virtual std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> calculateTransitionMatrix();
    virtual void generateTMatFileBatched(std::string basename);
    void generateTMatFileLowMemory(std::string basename);
    void generateTMatFile(std::string basename);
    void generateResetMapping(std::ofstream& file);
    void generateModelFile(std::string basename, double timestep_multiplier);

};

#endif