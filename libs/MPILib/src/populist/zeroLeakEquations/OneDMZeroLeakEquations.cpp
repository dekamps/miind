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
#include <MPILib/include/populist/zeroLeakEquations/OneDMZeroLeakEquations.hpp>
#include <MPILib/include/populist/AdaptiveHazard.hpp>

namespace MPILib {
namespace populist {


namespace {

	// TODO: possibly inline these functions later
	double Master
	(
		const double					y[],  
		int								i,
		const AdaptiveHazard&			hazard, 
		const parameters::OneDMInputSetParameter&	par
	)
	{
		int n_max            = par._n_max_bins;
		double g_max         = par._g_max;
		double q_expanded    = par._q_expanded;
//		double t_since_rebin = par._t_since_rebinning;
		double f_return      = 0;

		// we have index space and g-space and they are mapped 1-1. For the hazard we need g-space for P we need
		// index space

		// calculate negative contribution first
		double g = static_cast<double>(i)/static_cast<double>(n_max - 1)*g_max;
		double downstream = g - q_expanded;
		
		// if downstream is negative, don't bother; no density there
		if (downstream >= 0)
		{
			double delta_g = g_max/(n_max - 1);
			// calculate the number of bins (positive number; fraction) that downstream is from g
			double steps = q_expanded/delta_g;

			// calculate the boundaries
			int i_min = i - static_cast<int>(ceil(steps));
#ifndef NDEBUG
			int i_max = i - static_cast<int>(floor(steps));

			// if frac = 0, then interpolation point is close to i_max
			double frac = steps - static_cast<int>(floor(steps));
#endif
			assert( frac >= 0);
			assert( i_min >= 0 && i_max >= 0);
			assert( i_min == i_max - 1);
			assert( i_max < n_max);

			//TODO: leave interpolation out for sec

			double h = hazard(downstream);
			double p = y[i_min];

			f_return += h*p;
		}

		double h_neg = hazard(g);
		f_return -= h_neg*y[i];

		return f_return;
	}

    int    Meq(double t, const double y[], double f[],
           void *params)
     {
    	parameters::OneDMInputSetParameter par = *((parameters::OneDMInputSetParameter*)params);

	   // build the hazard function
	  AdaptiveHazard
		hazard
		(
			par._par_input._a,
			par._par_input._b
		);

	   for (Index i = 0; i < par._n_current_bins; i++ )
		   f[i] = Master
					(
						y,
						i,
						hazard,
						par
					);

	   for (Index j = par._n_current_bins; j < par._n_max_bins; j++)
		   f[j] = 0;

       return GSL_SUCCESS;
     }

}

OneDMZeroLeakEquations::OneDMZeroLeakEquations
(
	Number&						n_bins,
	std::valarray<Potential>&		state,
	Potential&					check_sum,
	SpecialBins&				bins,
	parameters::PopulationParameter&		par_pop,		//!< serves now mainly to communicate t_s
	parameters::PopulistSpecificParameter&	par_spec,
	Potential&					delta_v		//!< current potential interval covered by one bin, delta_v
):
AbstractZeroLeakEquations
(
	n_bins,
	state,
	check_sum,
	bins,
	par_pop,
	par_spec,
	delta_v
),
_n_bins(n_bins),
_p_state(&state),
_system(InitializeSystem()),
_convertor
(
	bins,
	par_pop,
	par_spec,
	delta_v,
	n_bins
),
_n_max(0),
_p_step(0),
_p_control(0),
_p_evolve(0)
{
}

OneDMZeroLeakEquations::~OneDMZeroLeakEquations()
{
	if (_p_step)
		gsl_odeiv_step_free (_p_step);
	if (_p_control)
		gsl_odeiv_control_free(_p_control);
	if (_p_evolve)
		gsl_odeiv_evolve_free (_p_evolve);
}

void OneDMZeroLeakEquations::Configure
(
	void*								p_par //OneDMParameter
)
{
	_n_max  = _convertor.PopSpecific().getMaxNumGridPoints();

	if (_p_step)
		gsl_odeiv_step_free (_p_step);
	if (_p_control)
		gsl_odeiv_control_free(_p_control);
	if (_p_evolve)
		gsl_odeiv_evolve_free (_p_evolve);

	_p_step    = gsl_odeiv_step_alloc (gsl_odeiv_step_rkf45, _n_max);
	_p_control = gsl_odeiv_control_y_new (1e-6, 0.0);
	_p_evolve  = gsl_odeiv_evolve_alloc (_n_max);

	_sys.function  = Meq;
	_sys.jacobian  = 0;
    _sys.dimension = _n_max;
	_sys.params    = static_cast<void*>(&_params);

	// the valarrays don't play a role but are required by the base class
	std::valarray<Potential> val;
	_convertor.Configure(val, val,*(static_cast<parameters::OneDMParameter*>(p_par)));

}

void OneDMZeroLeakEquations::Apply(Time time)
{
	_params = _convertor.InputSet();

	Time t = 0;
	Time h = time;
#ifndef NDEBUG
	int ires;
	ires =  
#endif
		gsl_odeiv_evolve_apply 
		(
			_p_evolve, 
			_p_control, 
			_p_step, 
			&_sys, 
			&t, 
			time, 
			&h, 
			&((*_p_state)[0])
		);

	assert(ires == GSL_SUCCESS);
}

gsl_odeiv_system OneDMZeroLeakEquations::InitializeSystem() const
{
	// in for historic reeasons
	gsl_odeiv_system sys_ret;
	sys_ret.dimension = 0;		// silence compiler warnings
	return sys_ret;
}

Rate OneDMZeroLeakEquations::CalculateRate() const
{
	// rate is the hazard weighted integral over the density
	// since the density profile can be quite ragged, simply calculate
	// the Riemann sum

	double delta_g = _convertor.ParPop()._theta/(_n_bins - 1);
	AdaptiveHazard 
		hazard
		(
			_params._par_input._a,
			_params._par_input._b
		);

	double sum = 0;
	for (Index i = 0; i < _n_bins; i++)
	{
		double g = delta_g*i;

		double rho = (*_p_state)[i];

		sum += rho*hazard(g);
	}
	return  sum;
}

} /* namespace populist */
} /* namespace MPILib */
