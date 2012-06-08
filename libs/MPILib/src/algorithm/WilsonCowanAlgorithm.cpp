/*
 * WilsonCowanAlgorithm.cpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/algorithm/WilsonCowanAlgorithm.hpp>

#include <NumtoolsLib/NumtoolsLib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

#include <functional>
#include <numeric>

namespace {

int sigmoid(double t, const double y[], double f[], void *params) {
	DynamicLib::WilsonCowanParameter* p_parameter =
			(DynamicLib::WilsonCowanParameter *) params;

	f[0] = (-y[0]
			+ p_parameter->_rate_maximum
					/ (1 + exp(-p_parameter->_f_noise * p_parameter->_f_input)))
			/ p_parameter->_time_membrane;

	return GSL_SUCCESS;
}

int sigmoidprime(double t, const double y[], double *dfdy, double dfdt[],
		void *params) {
	DynamicLib::WilsonCowanParameter* p_parameter =
			(DynamicLib::WilsonCowanParameter *) params;
	gsl_matrix_view dfdy_mat = gsl_matrix_view_array(dfdy, 1, 1);

	gsl_matrix * m = &dfdy_mat.matrix;

	gsl_matrix_set(m, 0, 0, -1 / p_parameter->_time_membrane);

	dfdt[0] = 0.0;

	return GSL_SUCCESS;

}
}
;

namespace MPILib {
namespace algorithm{

WilsonCowanAlgorithm::WilsonCowanAlgorithm() :
		AlgorithmInterface<double>(), _integrator(0, getInitialState(), 0, 0,
				NumtoolsLib::Precision(WC_ABSOLUTE_PRECISION,
						WC_RELATIVE_PRECISION), sigmoid, sigmoidprime) {
	// TODO Auto-generated constructor stub

}

WilsonCowanAlgorithm::WilsonCowanAlgorithm(const DynamicLib::WilsonCowanParameter&parameter) :
		AlgorithmInterface<double>(), _parameter(parameter), _integrator(0,
				getInitialState(), 0, 0,
				NumtoolsLib::Precision(WC_ABSOLUTE_PRECISION,
						WC_RELATIVE_PRECISION), sigmoid, sigmoidprime) {
	_integrator.Parameter() = _parameter;
}

WilsonCowanAlgorithm::~WilsonCowanAlgorithm() {
	// TODO Auto-generated destructor stub
}

WilsonCowanAlgorithm* WilsonCowanAlgorithm::clone() const {
	return new WilsonCowanAlgorithm(*this);
}

void WilsonCowanAlgorithm::configure(const DynamicLib::SimulationRunParameter& simParam) {

	NumtoolsLib::DVIntegratorStateParameter<DynamicLib::WilsonCowanParameter> parameter_dv;

	parameter_dv._vector_state = vector<double>(1, 0);
	parameter_dv._time_begin = simParam.TBegin();
	parameter_dv._time_end = simParam.TEnd();
	parameter_dv._time_step = simParam.TStep();
	parameter_dv._time_current = simParam.TBegin();

	parameter_dv._parameter_space = _parameter;

	parameter_dv._number_maximum_iterations =
			simParam.MaximumNumberIterations();

	_integrator.Reconfigure(parameter_dv);
//FIXME
}

void WilsonCowanAlgorithm::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {

	double f_inner_product = innerProduct(nodeVector, weightVector);

	_integrator.Parameter()._f_input = f_inner_product;

	try {
		while (_integrator.Evolve(time) < time)
			;
	} catch (NumtoolsLib::DVIntegratorException& except) {
		//FIXME
//		if (except.Code() == NumtoolsLib::NUMBER_ITERATIONS_EXCEEDED)
//			throw miind_parallel_fail(STR_NUMBER_ITERATIONS_EXCEEDED);
//		else
			throw except;
	}
}



Time WilsonCowanAlgorithm::getCurrentTime() const {
	return _integrator.CurrentTime();

}

Rate WilsonCowanAlgorithm::getCurrentRate() const {
	return *_integrator.BeginState();

}

double WilsonCowanAlgorithm::innerProduct(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector) {

	assert(nodeVector.size()==weightVector.size());

	if (nodeVector.begin() == nodeVector.end())
		return 0;

	return std::inner_product(nodeVector.begin(), nodeVector.end(),
			weightVector.begin(), 0.0);

}

vector<double> WilsonCowanAlgorithm::getInitialState() const {
	vector<double> array_return(WILSON_COWAN_STATE_DIMENSION);
	array_return[0] = 0;
	return array_return;
}

DynamicLib::AlgorithmGrid WilsonCowanAlgorithm::getGrid() const
{
	return _integrator.State();
}

} /* namespace algorithm */
} /* namespace MPILib */
