#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <TwoDLib/SimulationParserCPU.h>

// The entirety of MIIND is based on templated code which is not resolved when MIIND is built.
// It is resolved when the cpp code is generated in python and compiled by the user.
// This means that if we want to have a standalone executable or python library,
// we need to do some seriously rough hacking.
// As though we are eating roasted Ortolan, we must cover our heads in a veil to hide the shame
// from God.
// At least we know that there are only three possible Weight Types (In the future, we should only support
// CustomConnectionParameters as it is the most flexible). However, which model we need is not known
// until we look at WeightType in the XML file.
SimulationParserCPU<MPILib::CustomConnectionParameters>* modelCcp;
SimulationParserCPU<MPILib::DelayedConnection>* modelDc;
SimulationParserCPU<double>* modelDouble;

void InitialiseModel(int num_nodes, std::string filename) {
    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str())) {
        std::cout << "Failed to load XML simulation file.\n";
        return;
    }

    if (std::string("CustomConnectionParameters") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: CustomConnectionParameters.\n";
        modelCcp = new SimulationParserCPU<MPILib::CustomConnectionParameters>(num_nodes, filename);
        modelCcp->init();
    }
    else if (std::string("DelayedConnection") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: DelayedConnection.\n";
        modelDc = new SimulationParserCPU<MPILib::DelayedConnection>(num_nodes, filename);
        modelDc->init();
    }
    else if (std::string("double") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: double.\n";
        modelDouble = new SimulationParserCPU<double>(num_nodes, filename);
        modelDouble->init();
    }
}

void InitialiseModel(std::string filename) {
    InitialiseModel(1, filename);
}

PyObject* miind_init(PyObject* self, PyObject* args)
{
    if (modelCcp) {
        delete modelCcp;
        modelCcp = NULL;
    }

    if (modelDc) {
        delete modelDc;
        modelDc = NULL;
    }

    if (modelDouble) {
        delete modelDouble;
        modelDouble = NULL;
    }

    int nodes;
    char* filename;
    if (PyArg_ParseTuple(args, "s", &filename)) {
        InitialiseModel(std::string(filename));
    } else if (PyArg_ParseTuple(args, "is", &nodes, &filename)) {
        InitialiseModel(nodes, std::string(filename));
    } else
        return NULL;

    Py_RETURN_NONE;
}

PyObject* miind_getTimeStep(PyObject* self, PyObject* args)
{
    if (modelCcp)
        return Py_BuildValue("d", modelCcp->getTimeStep());
    if (modelDc)
        return Py_BuildValue("d", modelDc->getTimeStep());
    if (modelDouble)
        return Py_BuildValue("d", modelDouble->getTimeStep());
}

PyObject* miind_getSimulationLength(PyObject* self, PyObject* args)
{
    if (modelCcp)
        return Py_BuildValue("d", modelCcp->getSimulationLength());
    if (modelDc)
        return Py_BuildValue("d", modelDc->getSimulationLength());
    if (modelDouble)
        return Py_BuildValue("d", modelDouble->getSimulationLength());
}

PyObject* miind_startSimulation(PyObject* self, PyObject* args)
{
    if (modelCcp)
        modelCcp->startSimulation();
    else if (modelDc)
        modelDc->startSimulation();
    else if (modelDouble)
        modelDouble->startSimulation();

    Py_RETURN_NONE;
}

PyObject* miind_evolveSingleStep(PyObject* self, PyObject* args)
{
    PyObject* float_list;
    int pr_length;

    if (!PyArg_ParseTuple(args, "O", &float_list))
        return NULL;
    pr_length = PyObject_Length(float_list);
    if (pr_length < 0)
        return NULL;

    std::vector<double> activities(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject* item;
        item = PyList_GetItem(float_list, index);
        if (!PyFloat_Check(item))
            activities[index] = 0.0;
        activities[index] = PyFloat_AsDouble(item);
    }

    std::vector<double> out_activities;

    if (modelCcp)
        out_activities = modelCcp->evolveSingleStep(activities);
    else if (modelDc)
        out_activities = modelDc->evolveSingleStep(activities);
    else if (modelDouble)
        out_activities = modelDouble->evolveSingleStep(activities);

    PyObject* tuple = PyTuple_New(out_activities.size());

    for (int index = 0; index < out_activities.size(); index++) {
        PyTuple_SetItem(tuple, index, Py_BuildValue("d", out_activities[index]));
    }

    return tuple;
}

PyObject* miind_endSimulation(PyObject* self, PyObject* args)
{
    if (modelCcp)
        modelCcp->endSimulation();
    else if (modelDc)
        modelDc->endSimulation();
    else if (modelDouble)
        modelDouble->endSimulation();

    Py_RETURN_NONE;
}

/*
 * List of functions to add to miindsim in exec_miindsim().
 */
static PyMethodDef miindsim_functions[] = {
    {"init",  miind_init, METH_VARARGS, "Init Miind Model."},
    {"getTimeStep",  miind_getTimeStep, METH_VARARGS, "Get time step."},
    {"getSimulationLength",  miind_getSimulationLength, METH_VARARGS, "Get sim time."},
    {"startSimulation",  miind_startSimulation, METH_VARARGS, "Start simulation."},
    {"evolveSingleStep",  miind_evolveSingleStep, METH_VARARGS, "Evolve one time step."},
    {"endSimulation",  miind_endSimulation, METH_VARARGS, "Clean up."},
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for miindsim.
 */
PyDoc_STRVAR(miindsim_doc, "The miindsim module");

static PyModuleDef miindsim_def = {
    PyModuleDef_HEAD_INIT,
    "miindsim",
    miindsim_doc,
    -1,             
    miindsim_functions,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_miindsim() {
    return PyModule_Create(&miindsim_def);
}
