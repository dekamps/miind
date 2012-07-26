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
#include <cassert>
#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/populist/circulantSolvers/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/populist/AbstractRateComputation.hpp>
#include <MPILib/include/populist/circulantSolvers/CirculantSolver.hpp>
#include <MPILib/include/populist/zeroLeakEquations/NumericalZeroLeakEquations.hpp>
#include <MPILib/include/populist/nonCirculantSolvers/NonCirculantSolver.hpp>
#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>


namespace MPILib {
namespace populist {

int DerivReset(double t, const double y[], double dydt[], void * params){

	// the reset current is treated differently so that we know exactly what the refractive contribution is.

	parameters::NumericalZeroLeakParameter* p_param =
		static_cast<parameters::NumericalZeroLeakParameter*>(params);

	Number n_current = p_param->_nr_current_bins;
	Number n_max     = p_param->_nr_max_bins;
	double nu_e      = p_param->_eff_rate_exc;
	Index H          = p_param->_H;
	Index R          = p_param->_i_reset;
	double alpha_e   = p_param->_alpha_e;

	Index H_e = H;

	for (Index i = 0; i < n_current; i++)
		dydt[i] = 0;

	for (Index j = n_current; j < n_max; j++)
		dydt[j] = 0;


	// reset probability
	// special case 
	dydt[R] += nu_e*(1-alpha_e)*y[n_current-H_e-1];
	// rest
	for (Index i = n_current -H_e; i < n_current; i++ )
		dydt[R] += nu_e*y[i];

	return GSL_SUCCESS;
}

int Deriv(double t, const double y[], double dydt[], void * params){

	// The reset current is now treated differently because this is the part that will be refractive
	
	parameters::NumericalZeroLeakParameter* p_param =
		static_cast<parameters::NumericalZeroLeakParameter*>(params);

	Number n_current = p_param->_nr_current_bins;
	Number n_max     = p_param->_nr_max_bins;
	double nu_e      = p_param->_eff_rate_exc;
	double nu_i	     = p_param->_eff_rate_inh;
	Index H          = p_param->_H;
	double alpha_e   = p_param->_alpha_e;
	double alpha_i   = p_param->_alpha_i;
	double nu        = nu_e + nu_i;


	Index H_e = H;
	Index H_i = H;

	// did not use the right definition of alpha in the code below
	alpha_e = 1 - alpha_e;
	alpha_i = 1 - alpha_i;


	// first the diagonal; note that initialization of dydt is done here
	for (Index i = 0; i < H_i;i++)
		dydt[i] = -nu_e*y[i];

	dydt[H_i] = -(nu_e+alpha_i*nu_i)*y[H_i];

	for (Index i = H_i + 1; i < n_current; i++)
		dydt[i] = -nu*y[i];

	for (Index i = n_current; i < n_max; i++)
		dydt[i] = 0;

	// whereas here the changes only need to be added

	// lower triangle
	dydt[H_e] += nu_e*alpha_e*y[0];
	for (Index i = H_e + 1; i < n_current; i++){
		dydt[i] += nu_e*((1-alpha_e)*y[i-H_e-1] + alpha_e*y[i-H_e]);
	}


	// upper triangle
	dydt[n_current-H_i-1] += nu_i*alpha_i*y[n_current-1];
	for(Index i = 0; i < n_current-H_i-1;i++)
		dydt[i] += nu_i*(alpha_i*y[H_i+i] +(1-alpha_i)*y[H_i+i+1]);

	return GSL_SUCCESS;
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
_p_n_bins(&n_bins),
_p_par_pop(&par_pop),
_p_array_state(&array_state),
_p_check_sum(&check_sum),
_convertor
(
	VALUE_REF_INIT
	bins,
	par_pop,
	par_spec,
	delta_v,
	n_bins
)
{
	this->SetInputParameter(_convertor.getSolverParameter());
}

void NumericalZeroLeakEquations::InitializeIntegrators()
{
	Index i_reset = _convertor.getIndexCurrentResetBin();
	_p_integrator->Parameter()._alpha_e			= _convertor.getSolverParameter()._alpha_exc;
	_p_integrator->Parameter()._alpha_i			= _convertor.getSolverParameter()._alpha_inh;
	_p_integrator->Parameter()._eff_rate_exc	= _convertor.getSolverParameter()._rate_exc;
	_p_integrator->Parameter()._eff_rate_inh	= _convertor.getSolverParameter()._rate_inh;
	_p_integrator->Parameter()._H				= _convertor.getSolverParameter()._H_exc;
	_p_integrator->Parameter()._i_reset			=  i_reset;
	_p_integrator->Parameter()._nr_current_bins = *_p_n_bins;

	_p_reset->Parameter()._alpha_e			= _convertor.getSolverParameter()._alpha_exc;
	_p_reset->Parameter()._alpha_i			= _convertor.getSolverParameter()._alpha_inh;
	_p_reset->Parameter()._eff_rate_exc		= _convertor.getSolverParameter()._rate_exc;
	_p_reset->Parameter()._eff_rate_inh		= _convertor.getSolverParameter()._rate_inh;
	_p_reset->Parameter()._H				= _convertor.getSolverParameter()._H_exc;
	_p_reset->Parameter()._i_reset			=  i_reset;
	_p_reset->Parameter()._nr_current_bins  = *_p_n_bins;
	_p_reset->Parameter()._nr_current_bins	= *_p_n_bins;
	_p_reset->Parameter()._nr_max_bins		= this->ArrayState().size();
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

	Index i_reset = _convertor.getIndexCurrentResetBin();
	double before= (*_p_array_state)[i_reset];
	while( t_reset < time )
		t_reset      = _p_reset->Evolve(time);

	PushOnQueue	(time,before);
	PopFromQueue(time);

	while( t_integrator < time )
		t_integrator = _p_integrator->Evolve(time);
}

void NumericalZeroLeakEquations::PushOnQueue(Time t, double before)
{
	Index i_reset = _convertor.getIndexCurrentResetBin();
	double dif = (*_p_array_state)[i_reset] - before;
	(*_p_array_state)[i_reset] -= dif;

	StampedProbability prob;
	prob._prob = dif;
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
	_p_integrator = std::shared_ptr< ExStateDVIntegrator<parameters::NumericalZeroLeakParameter> >
					( 
						new ExStateDVIntegrator<parameters::NumericalZeroLeakParameter>
						(
							100000000,
							&((*_p_array_state)[0]),
							this->ParSpec().getMaxNumGridPoints(),
							1e-7,					// time step
							0.0,					// initial time
							NumtoolsLib::Precision(1e-5,	0.0),	// precision,
							Deriv
						)
					);

	_p_reset = std::shared_ptr< ExStateDVIntegrator<parameters::NumericalZeroLeakParameter> >
					( 
						new ExStateDVIntegrator<parameters::NumericalZeroLeakParameter>
						(
							100000000,
							&((*_p_array_state)[0]),
							this->ParSpec().getMaxNumGridPoints(),
							1e-7,					// time step
							0.0,					// initial time
							NumtoolsLib::Precision(1e-5,	0.0),	// precision,
							DerivReset
						)
					);

	parameters::InputParameterSet& input_set = _convertor.getSolverParameter();

	_p_rate_calc = auto_ptr<AbstractRateComputation>(this->ParSpec().getRateComputation().Clone());

	_p_rate_calc->Configure
	(
		*_p_array_state,
		input_set,
		_convertor.getParPop(),
		_convertor.getIndexReversalBin()
	);

	_p_integrator->Parameter()._nr_current_bins = *_p_n_bins;
	_p_integrator->Parameter()._nr_max_bins     = this->ParSpec().getMaxNumGridPoints();
	_p_integrator->Parameter()._i_reversal		= _convertor.getIndexReversalBin();
	_p_integrator->Parameter()._i_reset			= _convertor.getIndexCurrentResetBin();
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
	_convertor.SortConnectionvector(nodeVector, weightVector, typeVector);
}

void NumericalZeroLeakEquations::RecalculateSolverParameters()
{
	_convertor.RecalculateSolverParameters();
}

} /* namespace populist */
} /* namespace MPILib */
