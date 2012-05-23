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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "../NumtoolsLib/NumtoolsLib.h"
#include "ConnectionSquaredProduct.h"
#include "OrnsteinUhlenbeckAlgorithm.h"
#include "OrnsteinUhlenbeckParameter.h"
#include "PopulistException.h"
#include "Response.h"
#include "ResponseParameterBrunel.h"

using namespace PopulistLib;
using NumtoolsLib::Precision;

namespace {


	int OUResponse(double t, const double y[], double f[], void *params)
	{
		ResponseParameterBrunel* p_response_parameter
			= static_cast<ResponseParameterBrunel*>(params);

        f[0] = (-y[0] + ResponseFunction(*p_response_parameter) )/p_response_parameter->tau;

		return GSL_SUCCESS;
	}

}

namespace PopulistLib {

	OrnsteinUhlenbeckAlgorithm::OrnsteinUhlenbeckAlgorithm
	(
		istream& s
	):
	AbstractAlgorithm<OrnsteinUhlenbeckConnection>(0),
	_parameter_response(InitializeParameters(s)),
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

	OrnsteinUhlenbeckAlgorithm::OrnsteinUhlenbeckAlgorithm
	(
		const OrnsteinUhlenbeckParameter& parameter
	):
	AbstractAlgorithm<OrnsteinUhlenbeckConnection>(0),
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


	OrnsteinUhlenbeckAlgorithm::OrnsteinUhlenbeckAlgorithm
	(
		const OrnsteinUhlenbeckAlgorithm& algorithm
	):
	AbstractAlgorithm<OrnsteinUhlenbeckConnection>(algorithm.StateSize(algorithm.Grid())),
	_parameter_response(algorithm._parameter_response),
	_integrator(algorithm._integrator)
	{
	}

	OrnsteinUhlenbeckAlgorithm::~OrnsteinUhlenbeckAlgorithm()
	{
	}

	vector<double> OrnsteinUhlenbeckAlgorithm::InitialState() const
	{
		vector<double> array_return(OU_STATE_DIMENSION);
		array_return[0] = 0;
		return array_return;
	}

	ResponseParameterBrunel OrnsteinUhlenbeckAlgorithm::InitializeParameters
	(
		const OrnsteinUhlenbeckParameter& parameter
	) const
	{
		ResponseParameterBrunel parameter_brunel;
		parameter_brunel.tau			= parameter._tau;
		parameter_brunel.tau_refractive	= parameter._tau_refractive;
		parameter_brunel.V_reset        = parameter._V_reset;
		parameter_brunel.theta          = parameter._theta;


		return parameter_brunel;
	}

	bool OrnsteinUhlenbeckAlgorithm::Configure
	(
		const SimulationRunParameter& parameter_simulation
	)
	{
		NumtoolsLib::DVIntegratorStateParameter<ResponseParameterBrunel> parameter_dv;

		parameter_dv._vector_state = vector<double>(1,0);
		parameter_dv._time_begin   = parameter_simulation.TBegin();
		parameter_dv._time_end     = parameter_simulation.TEnd();
		parameter_dv._time_step    = parameter_simulation.TStep();
		parameter_dv._time_current = parameter_simulation.TBegin();

		parameter_dv._parameter_space = _parameter_response;
		parameter_dv._number_maximum_iterations = parameter_simulation.MaximumNumberIterations();

		_integrator.Reconfigure(parameter_dv);

		return true;
	}

	bool OrnsteinUhlenbeckAlgorithm::EvolveNodeState
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end,
		Time time_to_achieve
	)
	{
		_integrator.Parameter().mu = 
			static_cast<float>
			(
				_parameter_response.tau*InnerProduct
				(
					iter_begin, 
					iter_end
				)
			);
		_integrator.Parameter().sigma = 
			static_cast<float>
			(
				sqrt
				( 
					_parameter_response.tau*InnerSquaredProduct
					(
						iter_begin, 
						iter_end
					)
				) 
			);

		try
		{
			while( _integrator.Evolve(time_to_achieve) < time_to_achieve)
				;
		}
		catch(NumtoolsLib::DVIntegratorException except)
		{
			if (except.Code() == NumtoolsLib::NUMBER_ITERATIONS_EXCEEDED)
				throw DynamicLib::IterationNumberException(STRING_NUMBER_ITERATIONS_EXCEEDED);
			else 
				throw except;
		}
		
		return true;
	}

	OrnsteinUhlenbeckAlgorithm* OrnsteinUhlenbeckAlgorithm::Clone() const
	{
		return new OrnsteinUhlenbeckAlgorithm(*this);
	}

	bool OrnsteinUhlenbeckAlgorithm::Dump(ostream& s) const
	{
		return true;
	}

	AlgorithmGrid OrnsteinUhlenbeckAlgorithm::Grid() const
	{
		return _integrator.State();
	}

	AlgorithmGrid OrnsteinUhlenbeckAlgorithm::DefaultInitialGrid() const
	{
		vector<double> vector_grid(1,0);
		AlgorithmGrid grid(vector_grid);
		return grid;
	}

	NodeState OrnsteinUhlenbeckAlgorithm::State() const
	{
		vector<double> state(1);
		state[0] = *_integrator.BeginState();
		return state;
	}

	Time OrnsteinUhlenbeckAlgorithm::CurrentTime() const
	{
		return _integrator.CurrentTime();
	}

	Rate OrnsteinUhlenbeckAlgorithm::CurrentRate() const
	{
		return *_integrator.BeginState();
	}

	double OrnsteinUhlenbeckAlgorithm::InnerSquaredProduct
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end
	) const
	{
		Connection* p_begin = iter_begin.ConnectionPointer();
		Connection* p_end   = iter_end.ConnectionPointer();
		return inner_product
			(
				p_begin, 
				p_end,   
				p_begin, 
				0.0,
				plus<double>(),
				ConnectionSquaredProduct()
			);
	}

} // end of DelayActivityLib

string OrnsteinUhlenbeckAlgorithm::Tag() const
{
	return "<OrnsteinUhlenbeckAlgorithm>";
}

bool OrnsteinUhlenbeckAlgorithm::ToStream(ostream& s) const
{
	this->AbstractAlgorithm<OrnsteinUhlenbeckConnection>::ApplyBaseClassHeader(s,"OrnsteinUhlenbeckAlgorithm");

	s << this->InsertNameInTag(this->Tag(),this->GetName()) << "\n";
	OrnsteinUhlenbeckParameter par;
	par._tau			= this->_parameter_response.tau;
	par._tau_refractive	= this->_parameter_response.tau_refractive;
	par._theta			= this->_parameter_response.theta;
	par._V_reset		= this->_parameter_response.V_reset;
	par._V_reversal		= this->_parameter_response.V_reversal;
	par.ToStream(s);
	s << this->ToEndTag(this->Tag()) << "\n";

	this->AbstractAlgorithm<OrnsteinUhlenbeckConnection>::ApplyBaseClassFooter(s);

	return true;
}

bool OrnsteinUhlenbeckAlgorithm::FromStream(istream& s)
{
	return false;
}

ResponseParameterBrunel 
	OrnsteinUhlenbeckAlgorithm::InitializeParameters(istream& s) {
		ResponseParameterBrunel par_ret;
		string dummy;
		s >> dummy;

		if ( this->IsAbstractAlgorithmTag(dummy) ){
			s >> dummy;
			s >> dummy;
		}

		ostringstream str;
		str << dummy << ">";
		if ( str.str() != this->Tag() )
			throw PopulistException("OrnsteinUhlenbeckAlgorithm tag expected");

		getline(s,dummy);

		string name_alg;
		if (! this->StripNameFromTag(&name_alg, dummy) )
			throw PopulistException("OrnsteinUhlenbeckAlgorithm tag expected");
		this->SetName(name_alg);

		OrnsteinUhlenbeckParameter par_ou;
		par_ou.FromStream(s);

		par_ret.V_reset			= par_ou._V_reset;
		par_ret.V_reversal		= par_ou._V_reversal;
		par_ret.theta			= par_ou._theta;
		par_ret.tau				= par_ou._tau;
		par_ret.tau_refractive	= par_ou._tau_refractive;
		
		s >> dummy;
		if (dummy != this->ToEndTag(this->Tag()))
			throw PopulistException("OrnsteinUhlenbeck end tag expected");

		s >> dummy; // absorb AbstractAlgorithm tag

		return par_ret;
}