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
#include <MPILib/include/populist/IntegralRateComputation.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#include <UtilLib/IsFinite.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/StringDefinitions.hpp>

namespace MPILib {
namespace populist {
namespace {

gsl_spline* P_SPLINE;
gsl_interp_accel* P_ACCEL;

double density(double x, void* params = 0) {
	return gsl_spline_eval(P_SPLINE, x, P_ACCEL);
}
}

IntegralRateComputation::IntegralRateComputation() {
#ifndef DEBUG
	gsl_set_error_handler_off();
#endif
}

void IntegralRateComputation::Configure(std::valarray<Density>& array_state,
		const parameters::InputParameterSet& input_set,
		const parameters::PopulationParameter& par_population, Index index_reversal) {
	AbstractRateComputation::Configure(array_state, input_set, par_population,
			index_reversal);

	_p_accelerator = gsl_interp_accel_alloc();
	_p_workspace = gsl_integration_workspace_alloc(
			NUMBER_INTEGRATION_WORKSPACE);
	_array_interpretation.resize(_p_array_state->size());

}

IntegralRateComputation::~IntegralRateComputation() {
	if (_p_accelerator)
		gsl_interp_accel_free(_p_accelerator);
	if (_p_workspace)
		gsl_integration_workspace_free(_p_workspace);
}

Rate IntegralRateComputation::CalculateRate(Number nr_bins) {
	_n_bins = nr_bins;
	const parameters::InputParameterSet& _input_set = *_p_input_set;

	if (_input_set._rate_exc == 0)
		return 0.0;

	Potential v_cutoff = 1.0
			- _input_set._h_exc
					/ (_par_population._theta - _par_population._V_reversal);
	DefineRateArea(v_cutoff);

	if (_number_integration_area + 1 > NUMBER_INTEGRATION_WORKSPACE)
		throw utilities::Exception(WORKSPACE_EXCESSION);

	//TODO: flexible application of spline
	if (_number_integration_area < 4)
		P_SPLINE = gsl_spline_alloc(gsl_interp_linear,
				_number_integration_area);
	else
		P_SPLINE = gsl_spline_alloc(gsl_interp_akima, _number_integration_area);

	if (!P_SPLINE && _number_integration_area >= 4) {
		P_SPLINE = gsl_spline_alloc(gsl_interp_linear,
				_number_integration_area);
		std::cout << "Akima couldn't handle" << std::endl;
	}

	if (P_SPLINE == 0)
		throw utilities::Exception(
				"spline could not be defined in rate integral calculation");

	double* p_rho = &(_p_array_state->operator[](_start_integration_area));
	double* p_v = &_array_interpretation[_start_integration_area];

	gsl_spline_init(P_SPLINE, p_v, p_rho, _number_integration_area);

	void* dummy = 0;
	gsl_function F;
	F.function = density;
	F.params = dummy;

	double result, error;

	int error_code = gsl_integration_qags(&F, v_cutoff, 1.0, 1e-1, 0,
			NUMBER_INTEGRATION_WORKSPACE, _p_workspace, &result, &error);

	if (!::IsFinite(result)) {
		exit(1);
	}

	if (error_code == GSL_EROUND)
		throw utilities::Exception("GSL rounding error");

	if (error_code != GSL_SUCCESS)
		throw utilities::Exception("Integration fuck up");

	gsl_spline_free(P_SPLINE);

	return result * _input_set._rate_exc / _delta_v_rel;
}

IntegralRateComputation* IntegralRateComputation::Clone() const {
	return new IntegralRateComputation;
}
} /* namespace populist */
} /* namespace MPILib */
