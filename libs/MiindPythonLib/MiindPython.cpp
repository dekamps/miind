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

std::map<std::string, std::string> getVariablesFromFile(std::string filename) {
    std::map<std::string, std::string> dict;

    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str())) {
        std::cout << "Failed to load XML simulation file.\n";
        return dict;
    }
    

    for (pugi::xml_node var = doc.child("Simulation").child("Variable"); var; var = var.next_sibling("Variable")) {
        dict[std::string(var.attribute("Name").value())] = std::string(var.text().as_string());
    }

    return dict;
}

void InitialiseModel(int num_nodes, std::string filename, std::map<std::string,std::string> variables) {
    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str())) {
        std::cout << "Failed to load XML simulation file.\n";
        return;
    }

    if (std::string("CustomConnectionParameters") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: CustomConnectionParameters.\n";
        modelCcp = new SimulationParserCPU<MPILib::CustomConnectionParameters>(num_nodes, filename, variables);
        modelCcp->init();
    }
    else if (std::string("DelayedConnection") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: DelayedConnection.\n";
        modelDc = new SimulationParserCPU<MPILib::DelayedConnection>(num_nodes, filename, variables);
        modelDc->init();
    }
    else if (std::string("double") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: double.\n";
        modelDouble = new SimulationParserCPU<double>(num_nodes, filename, variables);
        modelDouble->init();
    }
}

void InitialiseModel(std::string filename, std::map<std::string, std::string> variables) {
    InitialiseModel(1, filename, variables);
}

std::map<std::string, std::string> ParseArguments(int& num_nodes, PyObject* args, PyObject* kwargs) {
    /* Get arbitrary number of strings from Py_Tuple */
    std::map<std::string, std::string> str_list;
    Py_ssize_t i = 0;
    PyObject *temp_p, *temp_p2;

    // First parameter is either an int to give the number of nodes...
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return str_list; }
    if (PyNumber_Check(temp_p) == 1) {
        /* Convert number to python long then C int*/
        temp_p2 = PyNumber_Long(temp_p);
        num_nodes = (int)PyLong_AsUnsignedLong(temp_p2);
        Py_DECREF(temp_p2);
        i++;
    }

    // ... or it's the xml filename
    temp_p = PyTuple_GetItem(args, i);
    if (temp_p == NULL) { return str_list; }
    if (PyUnicode_Check(temp_p) == 1) {
        PyObject* repr = PyObject_Repr(temp_p);
        PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
        const char* bytes = PyBytes_AS_STRING(str);
        std::string passed = std::string(bytes);
        // Bit of a weird key so nobody ends up creating a variable with the same name
        str_list[std::string("_1927482_MIIND_SIMULATION_FILENAME")] = passed.substr(1, passed.size() - 2);
        
        Py_XDECREF(repr);
        Py_XDECREF(str);
    }

    if (!kwargs)
        return str_list;

    // Now look at keyword args...
    PyObject* key, * value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        std::string str_ky;
        //PyObject* repr = PyObject_Repr(key);
        PyObject* str = PyUnicode_AsEncodedString(key, "utf-8", "~E~");
        const char* bytes = PyBytes_AS_STRING(str);

        str_ky = std::string(bytes);

        //Py_XDECREF(repr);
        Py_XDECREF(str);

        std::string str_val;
        PyObject* repr = PyObject_Repr(value);
        str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
        bytes = PyBytes_AS_STRING(str);

        str_val = std::string(bytes);

        Py_XDECREF(repr);
        Py_XDECREF(str);

        str_list[str_ky] = str_val.substr(1, str_val.size() - 2);
    }

    return str_list;
}


PyObject* miind_init(PyObject *self, PyObject *args, PyObject *keywds)
{
    try {
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

        int nodes = -1;

        std::map<std::string, std::string> argvars = ParseArguments(nodes, args, keywds);
        std::string filename = argvars[std::string("_1927482_MIIND_SIMULATION_FILENAME")];
        std::map<std::string, std::string> sim_variables = getVariablesFromFile(filename);

        // Replace sim_variable values with matching keys in argvars
        std::map<std::string, std::string>::iterator it;

        for (it = argvars.begin(); it != argvars.end(); it++)
        {
            if (it->first == std::string("_1927482_MIIND_SIMULATION_FILENAME"))
                continue;

            if (!sim_variables.count(it->first)) {
                std::cout << "Warning: Named argument [" << it->first << "] passed to init does not match any variables in " << filename << "\n";
                continue;
            }

            sim_variables[it->first] = it->second;
        }

        if (nodes > 0)
            InitialiseModel(nodes, filename, sim_variables);
        else
            InitialiseModel(filename, sim_variables);

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during init()");
        return NULL;
    }
}

PyObject* miind_getTimeStep(PyObject* self, PyObject* args)
{
    try {
        if (modelCcp)
            return Py_BuildValue("d", modelCcp->getTimeStep());
        if (modelDc)
            return Py_BuildValue("d", modelDc->getTimeStep());
        if (modelDouble)
            return Py_BuildValue("d", modelDouble->getTimeStep());
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during getTimeStep()");
        return NULL;
    }
    
}

PyObject* miind_getCurrentSimTime(PyObject* self, PyObject* args)
{
    try {
        if (modelCcp)
            return Py_BuildValue("d", modelCcp->getCurrentSimTime());
        if (modelDc)
            return Py_BuildValue("d", modelDc->getCurrentSimTime());
        if (modelDouble)
            return Py_BuildValue("d", modelDouble->getCurrentSimTime());
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during getCurrentSimTime()");
        return NULL;
    }

}

PyObject* miind_getSimulationLength(PyObject* self, PyObject* args)
{
    try {
        if (modelCcp)
            return Py_BuildValue("d", modelCcp->getSimulationLength());
        if (modelDc)
            return Py_BuildValue("d", modelDc->getSimulationLength());
        if (modelDouble)
            return Py_BuildValue("d", modelDouble->getSimulationLength());
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during getSimulationLength()");
        return NULL;
    } 
}

PyObject* miind_startSimulation(PyObject* self, PyObject* args)
{
    try {
        if (modelCcp)
            modelCcp->startSimulation();
        else if (modelDc)
            modelDc->startSimulation();
        else if (modelDouble)
            modelDouble->startSimulation();

        Py_RETURN_NONE;
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during startSimulation()");
        return NULL;
    }
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

    try{
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
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during evolveSingleStep()");
        return NULL;
    }
}

PyObject* miind_endSimulation(PyObject* self, PyObject* args)
{
    try {
        if (modelCcp)
            modelCcp->endSimulation();
        else if (modelDc)
            modelDc->endSimulation();
        else if (modelDouble)
            modelDouble->endSimulation();

        Py_RETURN_NONE;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled Exception during EndSimulation().");
        return NULL;
    }
}

/*
 * List of functions to add to miindsim in exec_miindsim().
 */
static PyMethodDef miindsim_functions[] = {
    {"init", (PyCFunction)miind_init, METH_VARARGS | METH_KEYWORDS, "Init Miind Model."},
    {"getTimeStep",  miind_getTimeStep, METH_VARARGS, "Get time step."},
    {"getCurrentSimulationTime",  miind_getCurrentSimTime, METH_VARARGS, "Get current sim time."},
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
