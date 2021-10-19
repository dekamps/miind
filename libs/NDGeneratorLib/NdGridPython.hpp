#ifndef MIINDGEN_GRID_PYTHON_HPP
#define MIINDGEN_GRID_PYTHON_HPP

#include "NdGrid.hpp"
#include "pyhelper.h"

class NdGridPython : public NdGrid {
public:
    std::string function_file_name;
    std::string function_name;

    CPyObject python_func;

    NdGridPython(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
        double _threshold_v, double _reset_v, double _timestep);
    ~NdGridPython();

    void setPythonFunctionFromStrings(std::string function, std::string functionname);
    void setPythonFunction(PyObject* function);

    NdCell generate_cell_with_coords(std::vector<unsigned int> cell_coord, bool btranslated) override;

    std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> calculateTransitionMatrix() override;
    void generateTMatFileBatched(std::string basename) override;
};

#endif