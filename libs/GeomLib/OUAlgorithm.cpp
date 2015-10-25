// Copyright (c) 2005 - 2015 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "OUAlgorithm.hpp"
#include "Response.hpp"

using namespace GeomLib;
using namespace GeomLib::algorithm;
using NumtoolsLib::Precision;

const int OU_STATE_DIMENSION       = 1;
const double OU_RELATIVE_PRECISION = 0;
const double OU_ABSOLUTE_PRECISION = 1e-5;

namespace {


	int OUResponse(double t, const double y[], double f[], void *params)
	{
		ResponseParameter* p_response_parameter
			= static_cast<ResponseParameter*>(params);

        f[0] = (-y[0] + ResponseFunction(*p_response_parameter) )/p_response_parameter->tau;

		return GSL_SUCCESS;
	}
}


	OUAlgorithm::OUAlgorithm
	(
		const NeuronParameter& parameter
	):
	  _parameter_neuron(parameter),
	  _parameter_response(InitializeParameters(parameter)),
	  _integrator
	  (
		0,
		InitialState(),
		0,
		0,
		Precision(OU_ABSOLUTE_PRECISION,OU_RELATIVE_PRECISION),
		OUResponse,
		0
	 )
	{
	}


	OUAlgorithm::OUAlgorithm
	(
		const OUAlgorithm& algorithm
	):
	  _parameter_neuron(algorithm._parameter_neuron),
	  _parameter_response(algorithm._parameter_response),
	  _integrator(algorithm._integrator)
	{
	}

	OUAlgorithm::~OUAlgorithm()
	{
	}

	vector<double> OUAlgorithm::InitialState() const
	{
		vector<double> array_return(OU_STATE_DIMENSION);
		array_return[0] = 0;
		return array_return;
	}

	ResponseParameter OUAlgorithm::InitializeParameters
	(
		const NeuronParameter& parameter
	) const
	{
		ResponseParameter  par;
		par.tau		   = parameter._tau;
		par.tau_refractive = parameter._tau_refractive;
		par.V_reset        = parameter._V_reset;
		par.theta          = parameter._theta;


		return par;
	}

	void OUAlgorithm::configure
	(
		const SimulationRunParameter& parameter_simulation
	)
	{
		NumtoolsLib::DVIntegratorStateParameter<ResponseParameter> parameter_dv;

		parameter_dv._vector_state = vector<double>(1,0);
		parameter_dv._time_begin   = parameter_simulation.getTBegin();
		parameter_dv._time_end     = parameter_simulation.getTEnd();
		parameter_dv._time_step    = parameter_simulation.getTStep();
		parameter_dv._time_current = parameter_simulation.getTBegin();

		parameter_dv._parameter_space = _parameter_response;
		parameter_dv._number_maximum_iterations = parameter_simulation.getMaximumNumberIterations();

		_integrator.Reconfigure(parameter_dv);
	}

void OUAlgorithm::evolveNodeState
(
 const std::vector<Rate>&                      nodeVector,
 const std::vector<MPILib::DelayedConnection>& weightVector, 
 Time                                          time
)
{
  MuSigma ms = _scalar_product.Evaluate(nodeVector,weightVector,time);
  _integrator.Parameter().mu    = ms._mu;
  _integrator.Parameter().sigma = ms._sigma;

  try
    {
      while( _integrator.Evolve(time) < time)
	;
    }
  catch(NumtoolsLib::DVIntegratorException except)
    {
      if (except.Code() == NumtoolsLib::NUMBER_ITERATIONS_EXCEEDED)
	throw GeomLibException("Number of iterations exceeded in GeomLib::OUAlgorithm");
      else 
	throw except;
    }
}

OUAlgorithm* OUAlgorithm::clone() const
{
  return new OUAlgorithm(*this);
}


AlgorithmGrid OUAlgorithm::getGrid() const
{
  return _integrator.State();
}

Time OUAlgorithm::getCurrentTime() const
{
  return _integrator.CurrentTime();
}

Rate OUAlgorithm::getCurrentRate() const
{
  return *_integrator.BeginState();
}

