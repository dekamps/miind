// Copyright (c) 2005 - 2010 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include "WilsonCowanAlgorithm.h"
#include "WilsonCowanParameter.h"
#include "IterationNumberException.h"
#include "StateConfigurationException.h"
#include "LocalDefinitions.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

using namespace DynamicLib;
using NumtoolsLib::Precision;

// unnamed namespace
namespace
{

	int sigmoid(double t, const double y[], double f[], void *params)
	{
		WilsonCowanParameter* p_parameter = (WilsonCowanParameter *)params;

		f[0] = (-y[0] + p_parameter->_rate_maximum/(1 + exp(-p_parameter->_f_noise*p_parameter->_f_input))) /p_parameter->_time_membrane;

		return GSL_SUCCESS;
	}


	int sigmoidprime(double t, const double y[], double *dfdy, double dfdt[], void *params)
	{
		WilsonCowanParameter* p_parameter = (WilsonCowanParameter *)params;
		gsl_matrix_view dfdy_mat  = gsl_matrix_view_array (dfdy, 1, 1);
 
		gsl_matrix * m = &dfdy_mat.matrix; 
	

		gsl_matrix_set (m, 0, 0, -1/p_parameter->_time_membrane); 

		dfdt[0] = 0.0;
	
		return GSL_SUCCESS;

	}
};

WilsonCowanAlgorithm::WilsonCowanAlgorithm
(
	istream& s
):
AbstractAlgorithm<double>(WILSON_COWAN_STATE_DIMENSION),
_parameter(InitializeParameter(s)),
_integrator
	(
		0,
		InitialState(),
		0,
		0,
		Precision(WC_ABSOLUTE_PRECISION,WC_RELATIVE_PRECISION),
		sigmoid,
		sigmoidprime
	)
{
}

WilsonCowanParameter WilsonCowanAlgorithm::InitializeParameter(istream& s)
{
	string dummy;

	s >> dummy;

	if ( this->IsAbstractAlgorithmTag(dummy) ){
		s >> dummy;
		s >> dummy;
	}

	ostringstream str;
	str << dummy << ">";
	if ( str.str() != this->Tag() )
		throw DynamicLibException("WilsonCowanAlgorithm tag expected");

	getline(s,dummy);

	string name_alg;
	if (! this->StripNameFromTag(&name_alg, dummy) )
			throw DynamicLibException("RateAlgorithm tag expected");

	this->SetName(name_alg);
	WilsonCowanParameter par;
	par.FromStream(s);
	this->StripFooter(s);

	return par;
}

void WilsonCowanAlgorithm::StripFooter(istream& s)
{
	string dummy;

	s >> dummy;

	if ( dummy != this->ToEndTag(this->Tag() ) )
		throw DynamicLibException("WilsonCowanAlgorithm end tag expected");

	// absorb the AbstractAlgorithm tag
	s >> dummy;

}


WilsonCowanAlgorithm::WilsonCowanAlgorithm
(
	const WilsonCowanParameter& parameter
):
AbstractAlgorithm<double>(WILSON_COWAN_STATE_DIMENSION),
_parameter(parameter),
_integrator
	(
		0,
		InitialState(),
		0,
		0,
		Precision(WC_ABSOLUTE_PRECISION,WC_RELATIVE_PRECISION),
		sigmoid,
		sigmoidprime
	)
{
	_integrator.Parameter() = _parameter;
}

WilsonCowanAlgorithm::WilsonCowanAlgorithm(const WilsonCowanAlgorithm& algorithm):
AbstractAlgorithm<double>(algorithm),
_parameter(algorithm._parameter),
_integrator(algorithm._integrator)
{
	_integrator.Parameter() = _parameter;
}

WilsonCowanAlgorithm::~WilsonCowanAlgorithm()
{
}

bool WilsonCowanAlgorithm::EvolveNodeState
(
	predecessor_iterator iter_begin,
	predecessor_iterator iter_end,
	Time time_to_achieve
)
{
	double f_inner_product = InnerProduct(iter_begin, iter_end);

	_integrator.Parameter()._f_input = f_inner_product;
	try
	{
		while( _integrator.Evolve(time_to_achieve) < time_to_achieve)
			;
	}
	catch(NumtoolsLib::DVIntegratorException except)
	{
		if (except.Code() == NumtoolsLib::NUMBER_ITERATIONS_EXCEEDED)
			throw IterationNumberException(STR_NUMBER_ITERATIONS_EXCEEDED);
		else 
			throw except;
	}
	return true;
}

vector<double> WilsonCowanAlgorithm::InitialState() const
{
	vector<double> array_return(WILSON_COWAN_STATE_DIMENSION);
	array_return[0] = 0;
	return array_return;
}

AlgorithmGrid WilsonCowanAlgorithm::Grid() const
{
	return _integrator.State();
}

string WilsonCowanAlgorithm::Tag() const
{
	return STRING_WC_TAG;
}

bool WilsonCowanAlgorithm::FromStream(istream& s)
{
	return false;
}

bool WilsonCowanAlgorithm::ToStream(ostream& s) const
{
	this->AbstractAlgorithm<double>::ApplyBaseClassHeader(s,"WilsonCowanAlgorithm");

	s << this->InsertNameInTag(this->Tag(),this->GetName())	<< "\n";
	_parameter.ToStream(s);
	s << ToEndTag(Tag())<< "\n";

	this->AbstractAlgorithm<double>::ApplyBaseClassFooter(s);

	return true;
}

WilsonCowanAlgorithm* WilsonCowanAlgorithm::Clone() const
{
	return new WilsonCowanAlgorithm(*this);
}

NodeState WilsonCowanAlgorithm::State() const
{
	vector<double> state(1);
	state[0] = *_integrator.BeginState();
	return state;
}


bool WilsonCowanAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	NumtoolsLib::DVIntegratorStateParameter<WilsonCowanParameter> parameter_dv;

	parameter_dv._vector_state = vector<double>(1,0);
	parameter_dv._time_begin   = parameter_simulation.TBegin();
	parameter_dv._time_end     = parameter_simulation.TEnd();
	parameter_dv._time_step    = parameter_simulation.TStep();
	parameter_dv._time_current = parameter_simulation.TBegin();
	
	parameter_dv._parameter_space = _parameter;

	parameter_dv._number_maximum_iterations = parameter_simulation.MaximumNumberIterations();

	_integrator.Reconfigure(parameter_dv);

	return true;
}

bool WilsonCowanAlgorithm::Dump(ostream& s) const
{
	return true;
}

Time WilsonCowanAlgorithm::CurrentTime() const
{
	return _integrator.CurrentTime();
}

Rate WilsonCowanAlgorithm::CurrentRate() const
{
	return *_integrator.BeginState();
}



WilsonCowanParameter WilsonCowanAlgorithm::Parameter() const
{
	return _parameter;
}
