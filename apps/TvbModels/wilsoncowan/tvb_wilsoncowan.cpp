#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <boost/timer/timer.hpp>
#include <GeomLib.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/report/handler/InactiveReportHandler.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/RateFunctorCode.hpp>
#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/WilsonCowanParameter.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/MiindTvbModelAbstract.hpp>

MPILib::Rate External_RateFunction(MPILib::Time t){
	return 1.0;
}
/* This class is designed to be used in conjunction with
 * Miind_WilsonCowan.py only. Any updates to this shared library
 * should be communicated to the TVB team and a new .so file should be provided
 * to be included in the tvb-library source tree (tvb-library/tvb/simulator/models/).
 */
class MiindWilsonCowan : public MPILib::MiindTvbModelAbstract<double, MPILib::utilities::CircularDistribution>{
private:
	std::vector<double> E_Initials = std::vector<double> (); // initial excitatory values
	std::vector<double> I_Initials = std::vector<double> (); // initial inhibitory values

public:

	MiindWilsonCowan(int num_nodes, double simulation_length, double dt) :
		MiindTvbModelAbstract(num_nodes, simulation_length) {
			this->_time_step = dt;
		}

	// In general, we won't need to set initial values but we implement it here
	// so that we can match TVB's Wilson Cowan example.
	void setInitialValues(std::vector<double> E_vals, std::vector<double> I_vals) {
		for(int i=0; i<_num_nodes; i++) {
			E_Initials.push_back(E_vals[i]);
			I_Initials.push_back(I_vals[i]);
		}
	}

	void init(std::vector<double> params)
	{
		// (TVB : tau_e) population time constant
		Time   E_tau       = params[4];
		// (TVB : c_e) maximum rate reached by the sigmoid function
		Rate   E_max_rate  = params[6];
		// (TVB : a_e) noise term modulates the input (a term in Wilson Cowan)
		double E_noise     = params[10];
		// (TVB : b_e) bias term shifts the input (theta term in Wilson Cowan)
		double E_bias      = params[12];
		// (TVB : r_e) smoothing term over output which models refractory dynamics
		double E_smoothing = params[8];
		// (TVB : tau_i) population time constant
		Time   I_tau       = params[5];
		// (TVB : c_i) maximum rate reached by the sigmoid function
		Rate   I_max_rate  = params[7];
		// (TVB : a_i) noise term modulates the input
		double I_noise     = params[11];
		// (TVB : b_i) bias term shifts the input
		double I_bias      = params[13];
		// (TVB : r_e) smoothing term over output which models refractory dynamics
		double I_smoothing = params[9];
		// (TVB : c_ee) Weight of E->E connection
		double E_E_Weight  = params[0];
		// (TVB : -c_ie) Weight of E->I connection
		double I_E_Weight  = -params[1];
		 // (TVB : c_ei) Weight of I->E connection
		double E_I_Weight  = params[2];
		// (TVB : -c_ii) Weight of I->I connection
		double I_I_Weight  = -params[3];
		// (TVB : P) Some additional drive to excitatory pop
		double P_E_Weight  = params[14];
		// (TVB : Q) Some additional drive to inhibitory pop
		double Q_I_Weight  = params[15];

		try {
			std::vector<NodeId> E_ids = std::vector<NodeId>();
			std::vector<NodeId> I_ids = std::vector<NodeId>();

			for(int i=0; i<_num_nodes; i++) {
				MPILib::WilsonCowanParameter E_param = MPILib::WilsonCowanParameter(
								E_tau, E_max_rate, E_noise, E_bias, E_Initials[i], E_smoothing);
				MPILib::WilsonCowanAlgorithm E_alg(E_param);

				MPILib::WilsonCowanParameter I_param = MPILib::WilsonCowanParameter(
								I_tau, I_max_rate, I_noise, I_bias, I_Initials[i], I_smoothing);
				MPILib::WilsonCowanAlgorithm I_alg(I_param);

				MPILib::Rate RateFunction_P(MPILib::Time);
				MPILib::RateFunctor<double> rate_functor_p(External_RateFunction);

				MPILib::Rate RateFunction_Q(MPILib::Time);
				MPILib::RateFunctor<double> rate_functor_q(External_RateFunction);

				MPILib::NodeId id_E = network.addNode(E_alg, MPILib::EXCITATORY_DIRECT);
				MPILib::NodeId id_I = network.addNode(I_alg, MPILib::INHIBITORY_DIRECT);
				MPILib::NodeId id_P = network.addNode(rate_functor_p, MPILib::NEUTRAL);
				MPILib::NodeId id_Q = network.addNode(rate_functor_q, MPILib::NEUTRAL);

				E_ids.push_back(id_E);
				I_ids.push_back(id_I);

				network.makeFirstInputOfSecond(id_E,id_E,E_E_Weight);
				network.makeFirstInputOfSecond(id_I,id_E,I_E_Weight);
				network.makeFirstInputOfSecond(id_E,id_I,E_I_Weight);
				network.makeFirstInputOfSecond(id_I,id_I,I_I_Weight);
				network.makeFirstInputOfSecond(id_P,id_E,P_E_Weight);
				network.makeFirstInputOfSecond(id_Q,id_I,Q_I_Weight);

			}

			// Set each node to have an external successor (for coupling input from TVB)
			for(auto& id : E_ids) {
				network.setNodeExternalSuccessor(id);
				network.setNodeExternalPrecursor(id, 1);
			}

			for(auto& id : I_ids) {
				network.setNodeExternalSuccessor(id);
				network.setNodeExternalPrecursor(id, 1);
			}

			std::string sim_name = "miind_wc";
			MPILib::report::handler::InactiveReportHandler handler =
											MPILib::report::handler::InactiveReportHandler();

			SimulationRunParameter par_run( handler,(_simulation_length/_time_step)+1,0,
											_simulation_length,_time_step,_time_step,sim_name,_time_step);

			network.configureSimulation(par_run);

		} catch(std::exception& exc){
			std::cout << exc.what() << std::endl;
		}
	}
};

static MiindWilsonCowan *model;

static PyObject *miind_init(PyObject *self, PyObject *args)
{
    int nodes;
    double sim_time;
    double time_step;

    if (!PyArg_ParseTuple(args, "idd", &nodes, &sim_time, &time_step))
	return NULL;
    
    model = new MiindWilsonCowan(nodes, sim_time, time_step);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *miind_initParams(PyObject *self, PyObject *args)
{
    PyObject *float_list;
    int pr_length;

    if (!PyArg_ParseTuple(args, "O", &float_list))
        return NULL;
    pr_length = PyObject_Length(float_list);
    if (pr_length < 0)
        return NULL;

    std::vector<double> params(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject *item;
        item = PyList_GetItem(float_list, index);
        if (!PyFloat_Check(item))
            params[index] = 0.0;
        params[index] = PyFloat_AsDouble(item);
    }

    model->init(params);
	
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *miind_setInitialValues(PyObject *self, PyObject *args)
{
    PyObject *float_listE;
    PyObject *float_listI;
    int pr_length;

    if (!PyArg_ParseTuple(args, "OO", &float_listE, &float_listI))
        return NULL;
    pr_length = PyObject_Length(float_listE);
    if (pr_length < 0)
        return NULL;

    std::vector<double> insE(pr_length);
    std::vector<double> insI(pr_length);

    for (int index = 0; index < pr_length; index++) {
        PyObject *item;
        item = PyList_GetItem(float_listE, index);
        if (!PyFloat_Check(item))
            insE[index] = 0.0;
        insE[index] = PyFloat_AsDouble(item);
    }

    for (int index = 0; index < pr_length; index++) {
        PyObject *item;
        item = PyList_GetItem(float_listI, index);
        if (!PyFloat_Check(item))
            insI[index] = 0.0;
        insI[index] = PyFloat_AsDouble(item);
    }

    model->setInitialValues(insE,insI);
	
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
    {"init",  miind_init, METH_VARARGS, "Init Miind model."},
    {"initParams",  miind_initParams, METH_VARARGS, "Init Miind model parameters."},
    {"getTimeStep",  miind_getTimeStep, METH_VARARGS, "Get time step."},
    {"getSimulationLength",  miind_getSimulationLength, METH_VARARGS, "Get sim time."},
    {"startSimulation",  miind_startSimulation, METH_VARARGS, "Start simulation."},
    {"evolveSingleStep",  miind_evolveSingleStep, METH_VARARGS, "Evolve one time step."},
    {"endSimulation",  miind_endSimulation, METH_VARARGS, "Clean up."},
    {"setInitialValues", miind_setInitialValues, METH_VARARGS, "Set initial values."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef miindmodule = {
    PyModuleDef_HEAD_INIT,
    "libmiindwc",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    MiindModelMethods
};

PyMODINIT_FUNC
PyInit_libmiindwc(void)
{
    return PyModule_Create(&miindmodule);
}
