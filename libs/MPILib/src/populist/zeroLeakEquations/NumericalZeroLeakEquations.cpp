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

#include <cassert>
#include <cassert>
#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/populist/parameters/NumericalZeroLeakParameter.hpp>
#include <MPILib/include/populist/zeroLeakEquations/NumericalZeroLeakEquations.hpp>
#include <MPILib/include/populist/rateComputation/IntegralRateComputation.hpp>
using namespace MPILib;
using namespace populist;
using namespace zeroLeakEquations;

namespace {

	int Deriv(double t, const double y[], double dydt[], void * params){

		// The reset current is now treated differently because this is the part that will be refractive

	        parameters::NumericalZeroLeakParameter* p_param = 
		    static_cast<parameters::NumericalZeroLeakParameter*>(params);

		Number n_current = p_param->_nr_current_bins;
		Number n_max     = p_param->_nr_max_bins;

		const std::vector<parameters::InputParameterSet>& vec_set = *(p_param->_p_vec_set);

		// initialize
		for (Index i = 0; i < n_max; i++)
			dydt[i] = 0;
		

		for (Index i_set = 0; i_set < vec_set.size(); i_set++){

			double nu_e      = vec_set[i_set]._rate_exc;
			double nu_i	     = vec_set[i_set]._rate_inh;


			Index H_e = vec_set[i_set]._H_exc;
			Index H_i = vec_set[i_set]._H_inh;

			assert(H_e < n_current);
			assert(H_i < n_current);

			double alpha_e   = vec_set[i_set]._alpha_exc;
			double alpha_i   = vec_set[i_set]._alpha_inh;

			// treat excitatory and inhibitory parts independently. Only do them if they have a non-zero rate.
			if (nu_i != 0.0){
					dydt[H_i] -= (1 - alpha_i)*nu_i*y[H_i];
					for (Index i_i = H_i + 1; i_i < n_current; i_i++)
						dydt[i_i] -= nu_i*y[i_i];

					Index H_i_diag = n_current - H_i - 1;
					for (Index i_i = 0; i_i < H_i_diag; i_i++)
						dydt[i_i] += nu_i*((1-alpha_i)*y[i_i + H_i] + alpha_i*y[i_i + H_i + 1]);
			
					dydt[H_i_diag] += nu_i*(1-alpha_i)*y[n_current-1];
			}

			if (nu_e != 0.0){
				for (Index i_e = 0; i_e < n_current; i_e++)
					dydt[i_e] -= nu_e*y[i_e];


				dydt[H_e] += ( 1 - alpha_e)*nu_e*y[0];
				for (Index i_e = H_e + 1; i_e < n_current; i_e++)
					dydt[i_e] += nu_e*(alpha_e*y[i_e-H_e-1] + (1 - alpha_e)*y[i_e-H_e]);

			}
		}
		return GSL_SUCCESS;
	}
	}

NumericalZeroLeakEquations::NumericalZeroLeakEquations
(
	Number&						n_bins,		
	valarray<Potential>&		array_state,
	Potential&					check_sum,
	SpecialBins&				bins,		
	parameters::PopulationParameter&		par_pop,		
	parameters::PopulistSpecificParameter&	par_spec,	
	Potential&					delta_v
):
AbstractZeroLeakEquations
(
	n_bins,
	array_state,
	check_sum,
	bins,
	par_pop,
	par_spec,
	delta_v
),
_time_current(0),
_p_n_bins(&n_bins),
_p_par_pop(&par_pop),
_p_array_state(&array_state),
_p_check_sum(&check_sum),
_convertor
(
	bins,
	par_pop,
	par_spec,
	delta_v,
	n_bins
)
{
}


void NumericalZeroLeakEquations::InitializeIntegrators()
{
	Index i_reset = _convertor.getIndexCurrentResetBin();
	_p_integrator->Parameter()._p_vec_set			= &_convertor.SolverParameter();
	_p_integrator->Parameter()._i_reset				=  i_reset;
	_p_integrator->Parameter()._nr_current_bins		= *_p_n_bins;
}

void NumericalZeroLeakEquations::Apply(Time time)
{

	InitializeIntegrators();

	// time is the time until the next bin is added.
	_time_current = _p_integrator->CurrentTime();

	// the integrator requires the total end time
	time += _time_current;
 

	Time t_integrator = 0;
	Time t_reset = 0;

	double before = _p_array_state->sum();
	while( t_integrator < time )
		t_integrator = _p_integrator->Evolve(time);

	double diff = before - _p_array_state->sum();
	PushOnQueue	(time,diff);
	PopFromQueue(time);

}

void NumericalZeroLeakEquations::PushOnQueue(Time t, double diff)
{
	StampedProbability prob;
	prob._prob = diff;
	prob._time = t + _p_par_pop->_tau_refractive;
	_queue.push(prob);
}

void NumericalZeroLeakEquations::PopFromQueue(Time t)
{
	Index i_reset = _convertor.getIndexCurrentResetBin();
	double p_pop = _queue.CollectAndRemove(t);
	(*_p_array_state)[i_reset] += p_pop;
}

void NumericalZeroLeakEquations::Configure
(
	void*
)
{
  _p_integrator = std::unique_ptr< ExStateDVIntegrator<parameters::NumericalZeroLeakParameter> >
					( 
					 new ExStateDVIntegrator<parameters::NumericalZeroLeakParameter>
						(
							100000000,
							&((*_p_array_state)[0]),
							this->ParSpec().getMaxNumGridPoints(),
							1e-6,					// time step
							0.0,					// initial time
							NumtoolsLib::Precision(1e-6,	0.0),	// precision,
							Deriv
						)
					);

        parameters::InputParameterSet& input_set = _convertor.getSolverParameter()[0];

    
	_p_rate_calc = std::unique_ptr<rateComputation::IntegralRateComputation>(new rateComputation::IntegralRateComputation);

	_p_rate_calc->Configure
	(
		*_p_array_state,
		_convertor.SolverParameter(),
		_convertor.getParPop(), 
		_convertor.getIndexReversalBin()
	);

	_p_integrator->Parameter()._nr_current_bins = *_p_n_bins;
	_p_integrator->Parameter()._nr_max_bins     = this->ParSpec().getMaxNumGridPoints();
	_p_integrator->Parameter()._i_reversal		= _convertor.getIndexReversalBin();
	_p_integrator->Parameter()._i_reset	        = _convertor.getIndexCurrentResetBin();
	_p_integrator->Parameter()._i_reset_orig	= _convertor.getIndexCurrentResetBin();
}


Rate NumericalZeroLeakEquations::CalculateRate() const
{
	return _p_rate_calc->CalculateRate(*_p_n_bins);
}

void NumericalZeroLeakEquations::AdaptParameters
(
)
{
	_convertor.AdaptParameters();
}

void NumericalZeroLeakEquations::SortConnectionvector
(
	const std::vector<Rate>& nodeVector,
	const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
	const std::vector<NodeType>& typeVector    
)
{
  _convertor.SortConnectionvector(nodeVector,weightVector,typeVector);
}

void NumericalZeroLeakEquations::RecalculateSolverParameters()
{
	_convertor.RecalculateSolverParameters();
}

