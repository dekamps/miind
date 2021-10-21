#include "NdGridPython.hpp"

NdGridPython::NdGridPython(std::vector<double> _base, std::vector<double> _dims, std::vector<unsigned int> _res,
    double _threshold_v, double _reset_v, std::vector<double> _reset_jump_relative, double _timestep) :
    NdGrid(_base, _dims, _res, _threshold_v, _reset_v, _reset_jump_relative, _timestep) {

        num_dimensions = _dims.size();

        generate_cell_coords(std::vector<unsigned int>(), resolution);
    }

NdGridPython::~NdGridPython() {
    Py_Finalize();
}

void NdGridPython::setPythonFunctionFromStrings(std::string function, std::string functionname) {

    function_file_name = function;
    function_name = functionname;

    Py_Initialize();

    CPyObject pName = PyUnicode_FromString(function_file_name.c_str());
    PyErr_Print();
    CPyObject pModule = PyImport_Import(pName);
    PyErr_Print();

    if (pModule)
    {
        python_func = PyObject_GetAttrString(pModule, function_name.c_str());
    }
    else
    {
        std::cout << "ERROR: Python module not imported\n";
    }
}

void NdGridPython::setPythonFunction(PyObject* func) {

    Py_Initialize();

    python_func.setObject(func);
}

NdCell NdGridPython::generate_cell_with_coords(std::vector<unsigned int> cell_coord, bool btranslated) {

    std::vector<double> base_point_coords(num_dimensions);
    for (unsigned int j = 0; j < num_dimensions; j++) {
        base_point_coords[j] = base[j] + (cell_coord[j] * (dimensions[j] / resolution[j]));
    }

    std::vector<NdPoint> ps = triangulator.generateUnitCubePoints(num_dimensions);
    for (unsigned int i = 0; i < ps.size(); i++) {

        PyObject* point_coord_list = PyList_New((Py_ssize_t)num_dimensions);

        std::vector<PyObject*> list_objects(num_dimensions);

        for (unsigned int d = 0; d < num_dimensions; d++) {
            ps[i].coords[d] *= (dimensions[d] / resolution[d]);
            ps[i].coords[d] += base_point_coords[d];

            list_objects[d] = PyFloat_FromDouble(ps[i].coords[d]);
            PyList_SetItem(point_coord_list, d, list_objects[d]);
        }

        //PyObject* time = PyFloat_FromDouble(0.0);
        //PyList_SetItem(point_coord_list, num_dimensions, time);

        PyObject* tuple = PyList_AsTuple(point_coord_list);
        if (btranslated) {
            if (python_func && PyCallable_Check(python_func))
            {
                PyObject* pass = Py_BuildValue("(O)", tuple);
                PyErr_Print();
                PyObject* pValue = PyObject_CallObject(python_func, pass);
                PyErr_Print();
                for (unsigned int d = 0; d < num_dimensions; d++) {
                    ps[i].coords[d] = ps[i].coords[d] + timestep * PyFloat_AsDouble(PyList_GetItem(pValue, d));
                }
            }
            else
            {
                std::cout << "ERROR: function.\n";
            }
        }
    }

    return NdCell(cell_coord, num_dimensions, ps, triangulator);
}

std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> NdGridPython::calculateTransitionMatrix() {
    std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> transitions;
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

void NdGridPython::generateTMatFileBatched(std::string basename) {
    unsigned int batch_size = 1000;
    std::ofstream file;
    file.open(basename + ".tmat");

    file << "0\t0\n";

    std::map<std::vector<unsigned int>, std::map<std::vector<unsigned int>, double>> transitions;
    for (unsigned int batch = 0; batch < (coord_list.size() / batch_size) + 1; batch++) {
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