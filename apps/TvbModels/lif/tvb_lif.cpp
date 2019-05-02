#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <TwoDLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>
#include <MPILib/include/DelayAlgorithmCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/MiindTvbModelAbstract.hpp>

/** This class is generically named MiindModel so no code change is required in TVB.
 * We can implement any simulation we desire and once the shared library is generated,
 * it can be copied to the TVB working directory and referenced when
 * instantiating the tvb.simulator.models.Miind class.
 *
 * e.g tvb.simulator.models.Miind('libmiindlif.so',76, 1**2, 0.001)
 */
class MiindModel : public MPILib::MiindTvbModelAbstract<DelayedConnection, MPILib::utilities::CircularDistribution> {
public:

	MiindModel(int num_nodes, double simulation_length) :
		MiindTvbModelAbstract(num_nodes, simulation_length){}

	void init()
	{
		_time_step = 0.001;
		for(int i=0; i<_num_nodes; i++) {
			std::vector<std::string> vec_mat_0{"lif_0.01_0_0_0_.mat"};
    	TwoDLib::MeshAlgorithm<DelayedConnection> alg_mesh_0("lif.model",vec_mat_0,_time_step);

      DelayedConnection con_external(2000,0.01,0);
			// As a basic example, we just implement a single lif node for each region
			// in TVB's connectivity.
			MPILib::NodeId id_0 = network.addNode(alg_mesh_0, MPILib::EXCITATORY_DIRECT);
			// Each node must have an incoming and outgoing connection to TVB
      network.setNodeExternalSuccessor(id_0);
			network.setNodeExternalPrecursor(id_0, con_external);
		}


		/* Any handler is permitted but as TVB provides it's own output, we get a speed
		 * increase by doing no IO in MIIND.
		 * If ROOT handlers are used, remember that the sim_name must include a
		 * directory and that directory must exist in the working directory.
		 * e.g sim_name = miind_output/miind_lif
		 */
		std::string sim_name = "miind_lif";
		report::handler::InactiveReportHandler handler = report::handler::InactiveReportHandler();

		SimulationRunParameter par_run( handler,(_simulation_length/_time_step)+1,0,
										_simulation_length,_time_step,_time_step,sim_name,_time_step);

		network.configureSimulation(par_run);
	}
};

static MiindModel *model;

static PyObject *miind_init(PyObject *self, PyObject *args)
{
    int nodes;
    double sim_time;

    if (!PyArg_ParseTuple(args, "ii", &nodes, &sim_time))
	return NULL;
    
    model = new MiindModel(nodes, sim_time);

    model->init();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *miind_getTimeStep(PyObject *self, PyObject *args)
{
    return Py_BuildValue("d", model->getTimeStep());
}

static PyObject *miind_getSimulationLength(PyObject *self, PyObject *args)
{
    return Py_BuildValue("d", model->getSimulationLength());
}

static PyObject *miind_startSimulation(PyObject *self, PyObject *args)
{
    model->startSimulation();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *miind_evolveSingleStep(PyObject *self, PyObject *args)
{
    PyObject *float_list;
    int pr_length;

    if (!PyArg_ParseTuple(args, "O", &float_list))
        return NULL;
    pr_length = PyObject_Length(float_list);
    if (pr_length < 0)
        return NULL;

    std::vector<double> activities(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject *item;
        item = PyList_GetItem(float_list, index);
        if (!PyFloat_Check(item))
            activities[index] = 0.0;
        activities[index] = PyFloat_AsDouble(item);
    }

    std::vector<double> out_activities = model->evolveSingleStep(activities);

    PyObject* tuple = PyTuple_New(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyTuple_SetItem(tuple, index, Py_BuildValue("d", out_activities[index]));
    }

    return tuple;
}

static PyObject *miind_endSimulation(PyObject *self, PyObject *args)
{
    model->endSimulation();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef MiindModelMethods[] = {
    {"init",  miind_init, METH_VARARGS, "Init Miind Model."},
    {"getTimeStep",  miind_getTimeStep, METH_VARARGS, "Get time step."},
    {"getSimulationLength",  miind_getSimulationLength, METH_VARARGS, "Get sim time."},
    {"startSimulation",  miind_startSimulation, METH_VARARGS, "Start simulation."},
    {"evolveSingleStep",  miind_evolveSingleStep, METH_VARARGS, "Evolve one time step."},
    {"endSimulation",  miind_endSimulation, METH_VARARGS, "Clean up."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef miindmodule = {
    PyModuleDef_HEAD_INIT,
    "libmiindlif",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    MiindModelMethods
};

PyMODINIT_FUNC
PyInit_libmiindlif(void)
{
    return PyModule_Create(&miindmodule);
}
