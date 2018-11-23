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
#ifndef _CODE_LIBS_GEOMLIB_NUMERICALMASTEREQUATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_NUMERICALMASTEREQUATIONCODE_INCLUDE_GUARD

#include "BasicDefinitions.hpp"
#include "CNZLCacheCode.hpp"
#include "NumericalMasterEquation.hpp"
#include "GeomLibException.hpp"

using namespace GeomLib;
using namespace MPILib;

namespace
{

    namespace
    {
        enum Type {EXCITATORY, INHIBITORY};
    }

    template <class Estimator>
    void HandleExcitatoryInput
	(	Rate 									        rate_e,
		const AbstractOdeSystem& 				        system,
		const vector<typename Estimator::CoverPair>&	vec_cover_pair,
		const double 							        y[],
		double 									        dydt[]
	)
    {
    	Number n_bins = system.NumberOfBins();
        Index i_reset  = system.IndexResetBin();

    	for (Index i = 0; i < n_bins; i++){

            typename Estimator::CoverPair pair_cover = vec_cover_pair[i];
            if (pair_cover.first._index == pair_cover.second._index) {
                 if (pair_cover.first._index >= 0 && pair_cover.first._index < static_cast<int>(n_bins)) {
                     Index i_trans = pair_cover.first._index;
                     double lower_bound = (1 - pair_cover.first._alpha);
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_e * (pair_cover.second._alpha - lower_bound) * y[system.MapPotentialToProbabilityBin(i_trans)];
                 }
             }

             if (pair_cover.first._index < pair_cover.second._index) {
                 if (pair_cover.first._index >= 0 && pair_cover.first._index < static_cast<int>(n_bins))
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_e * pair_cover.first._alpha * y[system.MapPotentialToProbabilityBin(pair_cover.first._index)];

                 if (pair_cover.second._index >= 0 && pair_cover.second._index < static_cast<int>(n_bins))
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_e * pair_cover.second._alpha * y[system.MapPotentialToProbabilityBin(pair_cover.second._index)];

                 for (int j = pair_cover.first._index + 1; j < pair_cover.second._index; j++)
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_e * y[system.MapPotentialToProbabilityBin(j)];
             }

             dydt[system.MapPotentialToProbabilityBin(i)] -= rate_e * y[system.MapPotentialToProbabilityBin(i)];

             assert(!(pair_cover.first._index > pair_cover.second._index));

             if (i == i_reset) {
            	  Rate rate = 0;
                  typename Estimator::CoverPair pair = vec_cover_pair[n_bins - 1];
                  rate += rate_e * (1 - pair.second._alpha) * y[system.MapPotentialToProbabilityBin(pair.second._index)];

                  for (Index ir = pair.second._index + 1; ir < n_bins; ir++)
                      rate += rate_e * y[system.MapPotentialToProbabilityBin(ir)];
                  // n_bins, does exist!
                  dydt[n_bins] += rate;
             }
    	}
    }


    template <class Estimator>
    void HandleInhibitoryInput
	(	Rate 			       	                     rate_i,
		const AbstractOdeSystem&                     system,
		const vector<typename Estimator::CoverPair>& vec_cover_pair,
		const double                                 y[],
		double                                       dydt[]
	)
    {
    	Number n_bins = system.NumberOfBins();

    	for (Index i = 0; i < n_bins; i++){
            typename Estimator::CoverPair pair_cover = vec_cover_pair[i];
            if (pair_cover.first._index == pair_cover.second._index) {
                 if (pair_cover.first._index >= 0 && pair_cover.first._index < static_cast<int>(n_bins)) {
                     Index i_trans = pair_cover.first._index;
                     double lower_bound = (1 - pair_cover.first._alpha);
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_i * (pair_cover.second._alpha - lower_bound) * y[system.MapPotentialToProbabilityBin(i_trans)];
                 }
             }

             if (pair_cover.first._index < pair_cover.second._index) {
                 if (pair_cover.first._index >= 0 && pair_cover.first._index < static_cast<int>(n_bins))
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_i * pair_cover.first._alpha * y[system.MapPotentialToProbabilityBin(pair_cover.first._index)];

                 if (pair_cover.second._index >= 0 && pair_cover.second._index < static_cast<int>(n_bins))
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_i * pair_cover.second._alpha * y[system.MapPotentialToProbabilityBin(pair_cover.second._index)];

                 for (int j = pair_cover.first._index + 1; j < pair_cover.second._index; j++)
                     dydt[system.MapPotentialToProbabilityBin(i)] += rate_i * y[system.MapPotentialToProbabilityBin(j)];
             }

             // the first index should never be larger than the second one, unless the second one = -1, which would not require any execution
             assert( pair_cover.second._index == -1 || (pair_cover.first._index <= pair_cover.second._index) );

             typename Estimator::CoverPair pair = vec_cover_pair[0];
             Index i_shift = static_cast<Index>(pair.first._index);
             //assert(i_shift >= 0);

             if (i == i_shift)
                 dydt[system.MapPotentialToProbabilityBin(i)] -= (rate_i * pair.first._alpha) * y[system.MapPotentialToProbabilityBin(i)];

             if (i > static_cast<Index>(i_shift))
                 dydt[system.MapPotentialToProbabilityBin(i)] -= rate_i * y[system.MapPotentialToProbabilityBin(i)];

    	}
    }

    static inline void SetDyDtZero(double dydt[], Number n_bins)
    {
    	// mind the equality sign
    	for (Index i = 0; i <=n_bins; i++)
    		dydt[i] = 0.0;
    }

  template <class Estimator>
  int CachedDeriv(double, const double y[], double dydt[], void* params)
  {
        MasterParameter<Estimator>* p_par = static_cast<MasterParameter<Estimator>* >(params);
    	const vector<InputParameterSet>& vec_set = *(p_par->_p_vec_set);

    	const AbstractOdeSystem& system = *(p_par->_p_system);
    	SetDyDtZero(dydt,system.NumberOfBins());

    	Index j = 0;
    	for(auto& set: vec_set){
    		if (set._rate_exc > 0)
    			HandleExcitatoryInput<Estimator>(set._rate_exc,  system, p_par->_p_cache->List()[j].first, y, dydt);
    		if (set._rate_inh > 0)
    			HandleInhibitoryInput<Estimator>(set._rate_inh,  system, p_par->_p_cache->List()[j].second, y, dydt);
    		j++;
    	}
    	return GSL_SUCCESS;
  }
}

template <class Estimator>
NumericalMasterEquation<Estimator>::NumericalMasterEquation
(
    AbstractOdeSystem& 	              	system,
    const DiffusionParameter&	       	par_diffusion,
    const CurrentCompensationParameter&	par_current
):
    _system(system),
    _par_diffusion(par_diffusion),
    _par_current(par_current),
    _estimator
    (
     _system.InterpretationBuffer(),
     _system.Par()
    ),
    _convertor
    (
        _system.Par()._par_pop,
        par_diffusion,
        par_current,
        _system.InterpretationBuffer()
    ),
    _integrator
    (
        MAXITERATION,
        &_system.MassBuffer()[0],
        _system.NumberOfBins() + 1, // TODO: review, rather hacky
        T_STEP,
        T_START,
        PRECISION,
        CachedDeriv<Estimator>
    ),
	_queue(TIME_QUEUE_BATCH),
	_rate(0.0),
	_scratch_dense(_system.NumberOfBins()),
	_scratch_pot(_system.NumberOfBins()+1)
{
}

template <class Estimator>
NumericalMasterEquation<Estimator>::~NumericalMasterEquation()
{
}

template <class Estimator>
void NumericalMasterEquation<Estimator>::sortConnectionVector
(
	const std::vector<MPILib::Rate>&	     	    vec_rates,
	const std::vector<MPILib::DelayedConnection>&	vec_cons,
	const std::vector<MPILib::NodeType>&	       	vec_types

)
{
    _convertor.SortConnectionvector(vec_rates,vec_cons,vec_types);
}

template <class Estimator>
void NumericalMasterEquation<Estimator>::InitializeIntegrator()
{
    _integrator.Parameter()._nr_bins    	= _system.NumberOfBins();
    _integrator.Parameter()._p_system		= &_system;
    _integrator.Parameter()._p_estimator	= &_estimator;
    _integrator.Parameter()._i_reset        = _system.IndexResetBin();
    _integrator.Parameter()._p_vec_set      = &_convertor.SolverParameter();
    _integrator.Parameter()._p_cache        = &_cache;
}

template <class Estimator>
double NumericalMasterEquation<Estimator>::RecaptureProbability()
{
	// existence of the extra bin is guaranteed in AbstractOdeSytem
	MPILib::Probability p = _system.MassBuffer()[_system.NumberOfBins()];

	_system.MassBuffer()[_system.NumberOfBins()] = 0.0;
	return p;
}

template <class Estimator>
void NumericalMasterEquation<Estimator>::apply(Time t)
{
    InitializeIntegrator();
    // initialize caching values for the bin estimator

    _cache.Initialize(_system, _convertor.SolverParameter(), _estimator);

    MPILib::Time t_integrator = 0;

    // time is the time until the next bin is added.
    _time_current = _integrator.CurrentTime();
    // the integrator requires the total end time
    t += _time_current;

    t_integrator = 0.0;

    double p = 0;

    while (t_integrator < t){
        t_integrator = _integrator.Evolve(t);
        p += RecaptureProbability();
    }


    MPILib::populist::StampedProbability prob;
    prob._prob = p;
    prob._time = t + _system.Par()._par_pop._tau_refractive;
    _queue.push(prob);
    _rate = p/_system.TStep();

    p = _queue.CollectAndRemove(t);
    _system.MassBuffer()[_system.MapPotentialToProbabilityBin(_system.IndexResetBin())] += p;
}

template <class Estimator>
Rate NumericalMasterEquation<Estimator>::getTransitionRate() const
{
    return  _rate;
}

template <class Estimator>
Rate NumericalMasterEquation<Estimator>::IntegralRate() const
{
	Potential v_reset = _system.Par()._par_pop._theta;
	Potential h = _convertor.SolverParameter()[0]._h_exc;
	Potential v_bound = v_reset - h;

	Number n_bins = _system.NumberOfBins();
	vector<Potential>& array_interpretation = _system.InterpretationBuffer();
	vector<Potential>& array_mass           = _system.MassBuffer();

	int i_bound = 0;
	for (int i = static_cast<int>(n_bins - 1); i >= 0; i--)
		if (array_interpretation[i] < v_bound){
			i_bound = i;
			break;
		}

	if (i_bound == static_cast<int>(n_bins - 1)){
		return 0.0;
	}

	_scratch_pot[i_bound] = v_bound;
	for (Index i = i_bound + 1; i < n_bins; i++)
		_scratch_pot[i] = array_interpretation[i];

	double integ = 0.0;
	_scratch_pot[n_bins] = v_reset;

	for (Index i = i_bound; i < n_bins; i++)
		_scratch_dense[i] = array_mass[_system.MapPotentialToProbabilityBin(i)]/(_scratch_pot[i+1]-_scratch_pot[i]);

	for (Index i = i_bound; i < n_bins-1; i++)
		integ += 0.5*(_scratch_dense[i]+_scratch_dense[i+1])*(_scratch_pot[i+1]-_scratch_pot[i]);
	integ += 0.5*_scratch_dense[n_bins-1]*(_scratch_pot[n_bins]-_scratch_pot[n_bins-1]);

	Rate rate = _convertor.SolverParameter()[0]._rate_exc;

	return rate*integ;

}

template <class Estimator>
Density NumericalMasterEquation<Estimator>::Checksum() const
{
	return std::accumulate(_system.MassBuffer().begin(),_system.MassBuffer().end(),0.0);
}
#endif
