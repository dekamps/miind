// Copyright (c) 2005 - 2011 Marc de Kamps
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

#include "../DynamicLib/DynamicLib.h"
#include "ClamLibException.h"
#include "SemiSigmoid.h"
#include "LocalDefinitions.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

using namespace std;
using ClamLib::SemiSigmoid;
using DynamicLib::AbstractAlgorithm;
using DynamicLib::IterationNumberException;
using DynamicLib::STR_NUMBER_ITERATIONS_EXCEEDED;
using DynamicLib::WILSON_COWAN_STATE_DIMENSION;
using DynamicLib::STRING_WC_TAG;
using DynamicLib::Time;
using NumtoolsLib::Precision;

// unnamed namespace
namespace
{

	int semisigmoid(double t, const double y[], double f[], void *params)
	{
		// from the point of view of DynamicLib, this might look like code replication, since 
		// there is a function sigmoid there, which does almost the same. The difference, however,
		// is input < 0, and GSL really requires a different function for its interface
		WilsonCowanParameter* p_parameter = (WilsonCowanParameter *)params;

		double semi_sig = (p_parameter->_f_input  > 0 ) ? \
			p_parameter->_rate_maximum*(2.0/(1 + exp(-p_parameter->_f_noise*p_parameter->_f_input))-1.0 ) : \
			0;

		f[0] = (-y[0] + semi_sig ) /p_parameter->_time_membrane;

		return GSL_SUCCESS;
	}


	int semisigmoidprime(double t, const double y[], double *dfdy, double dfdt[], void *params)
	{
		WilsonCowanParameter* p_parameter = (WilsonCowanParameter *)params;
		gsl_matrix_view dfdy_mat  = gsl_matrix_view_array (dfdy, 1, 1);
 
		gsl_matrix * m = &dfdy_mat.matrix; 
	

		gsl_matrix_set (m, 0, 0, -1/p_parameter->_time_membrane); 

		dfdt[0] = 0.0;
	
		return GSL_SUCCESS;

	}
};

SemiSigmoid::SemiSigmoid
(
	istream& s
):
AbstractAlgorithm<double>(SEMI_SIGMOID_STATE_DIMENSION),
_parameter(WilsonCowanParameterFromStream(s)),
_integrator
	(
		0,
		InitialState(),
		0,
		0,
		Precision(SS_ABSOLUTE_PRECISION,SS_RELATIVE_PRECISION),
		semisigmoid,
		semisigmoidprime
	)
{
  _integrator.Parameter() = _parameter;
  this->StripFooter(s);
}


SemiSigmoid::SemiSigmoid
(
	const WilsonCowanParameter& parameter
):
AbstractAlgorithm<double>(SEMI_SIGMOID_STATE_DIMENSION),
_parameter(parameter),
_integrator
	(
		0,
		InitialState(),
		0,
		0,
		Precision(SS_ABSOLUTE_PRECISION,SS_RELATIVE_PRECISION),
		semisigmoid,
		semisigmoidprime
	)
{
  _integrator.Parameter() = _parameter;
}

WilsonCowanParameter SemiSigmoid::WilsonCowanParameterFromStream(istream& s)
{
	WilsonCowanParameter par;
	this->StripHeader(s);
	string dummy;
	s >> dummy;
	this->SetName(this->UnWrapTag(dummy));
	par.FromStream(s);
	return par;
}

SemiSigmoid::SemiSigmoid(const SemiSigmoid& algorithm):
AbstractAlgorithm<double>(algorithm.StateSize(algorithm.Grid())),
_parameter(algorithm._parameter),
_integrator(algorithm._integrator)
{
  _integrator.Parameter();
}


SemiSigmoid::~SemiSigmoid()
{
}

bool SemiSigmoid::EvolveNodeState
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

vector<double> SemiSigmoid::InitialState() const
{
	vector<double> array_return(WILSON_COWAN_STATE_DIMENSION);
	array_return[0] = 0;
	return array_return;
}

AlgorithmGrid SemiSigmoid::Grid() const
{
	return _integrator.State();
}

string SemiSigmoid::Tag() const
{
	return STR_SS_TAG;
}

bool SemiSigmoid::FromStream(istream& s)
{
	return false;
}

void SemiSigmoid::StripHeader(istream& s)
{
	string dummy;

	s >> dummy;
	// Either an AbstractAlgorithm tag and then a PopulationAlgorithm tag, or just a PopulationAlgorithm tag when the stream already has been processed by a builder

	if ( this->IsAbstractAlgorithmTag(dummy) ){
		s >> dummy;
		s >> dummy;
	}

	if ( dummy != this->Tag() )
		throw ClamLibException("SemiSigmoid tag expected");
}

void SemiSigmoid::StripFooter(istream& s)
{
	string dummy;

	s >> dummy;

	if ( dummy != this->ToEndTag(this->Tag() ) )
		throw ClamLibException("SemiSigmoid end tag expected");

		// absorb the AbstractAlgorithm tag
	s >> dummy;

}

bool SemiSigmoid::ToStream(ostream& s) const
{
	this->AbstractAlgorithm<double>::ApplyBaseClassHeader(s,"SemiSigmoid");
	s << this->Tag()						<< "\n";
	s << WrapTag(this->GetName(),"name")	<< "\n";	
	_parameter.ToStream(s);
	s << this->ToEndTag(this->Tag())		<< "\n";
	this->AbstractAlgorithm<double>::ApplyBaseClassFooter(s);
	return true;
}

SemiSigmoid* SemiSigmoid::Clone() const
{
	return new SemiSigmoid(*this);
}

NodeState SemiSigmoid::State() const
{
	vector<double> state(1);
	state[0] = *_integrator.BeginState();
	return state;
}


bool SemiSigmoid::Configure
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

bool SemiSigmoid::Dump(ostream& s) const
{
	return true;
}

Time SemiSigmoid::CurrentTime() const
{
	return _integrator.CurrentTime();
}

Rate SemiSigmoid::CurrentRate() const
{
	return *_integrator.BeginState();
}

WilsonCowanParameter SemiSigmoid::Parameter() const
{
	return _parameter;
}
