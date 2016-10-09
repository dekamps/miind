// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/algorithm/WilsonCowanAlgorithm.hpp>
#include <MPILib/include/StringDefinitions.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

#include <NumtoolsLib/NumtoolsLib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

#include <functional>
#include <numeric>

namespace {

int sigmoid(double, const double y[], double f[], void *params) {
	auto p_parameter = (MPILib::algorithm::WilsonCowanParameter *) params;

	f[0] = (-y[0]
			+ p_parameter->_rate_maximum
					/ (1 + exp(-p_parameter->_f_noise * p_parameter->_f_input - p_parameter->_f_bias)))
			/ p_parameter->_time_membrane;

	return GSL_SUCCESS;
}

int sigmoidprime(double , const double[], double *dfdy, double dfdt[],
		void *params) {
	auto p_parameter = (MPILib::algorithm::WilsonCowanParameter *) params;
	gsl_matrix_view dfdy_mat = gsl_matrix_view_array(dfdy, 1, 1);

	gsl_matrix * m = &dfdy_mat.matrix;

	gsl_matrix_set(m, 0, 0, -1 / p_parameter->_time_membrane);

	dfdt[0] = 0.0;

	return GSL_SUCCESS;

}
}


namespace MPILib {
namespace algorithm {

WilsonCowanAlgorithm::WilsonCowanAlgorithm() :
		AlgorithmInterface<double>(), _integrator(0, getInitialState(), 0, 0,
				NumtoolsLib::Precision(WC_ABSOLUTE_PRECISION,
						WC_RELATIVE_PRECISION), sigmoid, sigmoidprime) {

}

WilsonCowanAlgorithm::WilsonCowanAlgorithm(const WilsonCowanParameter&parameter) :
		AlgorithmInterface<double>(), _parameter(parameter), _integrator(0,
				getInitialState(), 0, 0,
				NumtoolsLib::Precision(WC_ABSOLUTE_PRECISION,
						WC_RELATIVE_PRECISION), sigmoid, sigmoidprime) {
	_integrator.Parameter() = _parameter;
}

WilsonCowanAlgorithm::~WilsonCowanAlgorithm() {
}

WilsonCowanAlgorithm* WilsonCowanAlgorithm::clone() const {
	return new WilsonCowanAlgorithm(*this);
}

void WilsonCowanAlgorithm::configure(const SimulationRunParameter& simParam) {

	NumtoolsLib::DVIntegratorStateParameter<WilsonCowanParameter> parameter_dv;

	parameter_dv._vector_state = std::vector<double>(1, 0);
	parameter_dv._time_begin = simParam.getTBegin();
	parameter_dv._time_end = simParam.getTEnd();
	parameter_dv._time_step = simParam.getTStep();
	parameter_dv._time_current = simParam.getTBegin();

	parameter_dv._parameter_space = _parameter;

	parameter_dv._number_maximum_iterations =
			simParam.getMaximumNumberIterations();

	_integrator.Reconfigure(parameter_dv);
}

void WilsonCowanAlgorithm::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {

	double f_inner_product = innerProduct(nodeVector, weightVector);

	_integrator.Parameter()._f_input = f_inner_product;

	try {
		while (_integrator.Evolve(time) < time)
			;
	} catch (NumtoolsLib::DVIntegratorException& except) {
		if (except.Code() == NumtoolsLib::NUMBER_ITERATIONS_EXCEEDED)
			throw utilities::ParallelException(STR_NUMBER_ITERATIONS_EXCEEDED);
		else
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

std::vector<double> WilsonCowanAlgorithm::getInitialState() const {
	std::vector<double> array_return(WILSON_COWAN_STATE_DIMENSION);
	array_return[0] = 0;
	return array_return;
}

AlgorithmGrid WilsonCowanAlgorithm::getGrid(NodeId) const {
	return _integrator.State();
}

} /* namespace algorithm */
} /* namespace MPILib */
