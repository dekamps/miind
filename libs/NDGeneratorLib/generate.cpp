#define PY_SSIZE_T_CLEAN
#include "NdGridPython.hpp"

void ParseArguments(PyObject* args) {
    /* Get arbitrary number of strings from Py_Tuple */
    Py_ssize_t i = 0;
    PyObject *temp_p, *temp_p2;

    PyObject* python_function;
    std::string basename;
    std::vector<double> base;
    std::vector<double> size;
    std::vector<unsigned int> resolution;
    double threshold = 0.0;
    double reset = 0.0;
    std::vector<double> relative_jump;
    double timestep = 0.0;
    double timescale = 0.0;

    // First argument is the python function
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyCallable_Check(temp_p) == 1) {
        python_function = temp_p;
        i++;
    }

    // Second argument is the basename
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyUnicode_Check(temp_p) == 1) {
        PyObject* str = PyUnicode_AsEncodedString(temp_p, "utf-8", "~E~");
        const char* bytes = PyBytes_AS_STRING(str);
        std::string passed = std::string(bytes);

        basename = passed;

        Py_XDECREF(str);
        i++;
    }

    // Third argument is the base list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    int pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    base = std::vector<double>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            base[index] = 0.0;
        base[index] = PyFloat_AsDouble(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Fourth argument is the grid size list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return ; }
    pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    size = std::vector<double>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            size[index] = 0.0;
        size[index] = PyFloat_AsDouble(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Fifth argument is the grid resolution list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    resolution = std::vector<unsigned int>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            resolution[index] = 0.0;
        resolution[index] = PyLong_AsLong(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Sixth argument is the threshold
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        threshold = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Seventh argument is the reset
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        reset = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Eighth argument is the relative jump list
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    pr_length = PyObject_Length(temp_p);
    if (pr_length < 0)
        return;

    relative_jump = std::vector<double>(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(temp_p, index);
        if (!PyFloat_Check(item))
            relative_jump[index] = 0.0;
        relative_jump[index] = PyFloat_AsDouble(item);
    }

    Py_XDECREF(temp_p);
    i++;

    // Nineth argument is the time step
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        timestep = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // Tenth argument is the time scale
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python float then C double*/
        temp_p2 = PyNumber_Float(temp_p);
        timescale = (double)PyFloat_AsDouble(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    NdGridPython g(base, size, resolution, threshold, reset, relative_jump, timestep);

    g.setPythonFunction(python_function);
    g.generateModelFile(basename, timescale);
    g.generateTMatFileBatched(basename);

}


PyObject* miind_generateNdGrid(PyObject *self, PyObject *args)
{
    try {
        ParseArguments(args);

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during generateNdGrid()");
        return NULL;
    }
}

/*
 * List of functions to add to miindsim in exec_miindsim().
 */
static PyMethodDef miindgen_functions[] = {
    {"generateNdGrid", (PyCFunction)miind_generateNdGrid, METH_VARARGS, "Generate an ND grid for use in MIIND."},
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for miindsim.
 */
PyDoc_STRVAR(miindgen_doc, "The miindgen module");

static PyModuleDef miindgen_def = {
    PyModuleDef_HEAD_INIT,
    "miindgen",
    miindgen_doc,
    -1,             
    miindgen_functions,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_miindgen() {
    return PyModule_Create(&miindgen_def);
}
