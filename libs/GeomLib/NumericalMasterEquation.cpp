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
#include "BasicDefinitions.hpp"
#include "NumericalMasterEquation.h"
#include "GeomLibException.hpp"

using namespace GeomLib;


namespace
{

    namespace
    {
        enum Type {EXCITATORY, INHIBITORY};
    }
    void AddCachedToRow
    (
        double                  dydt[],
        Index                   i,
        const double            y[],
        const AbstractOdeSystem& system,
        const BinEstimator&     estimator,
        Index                   j,
        Potential               h,
        Rate                    rate,
        const CNZLCache&        cache,
        Type                    type
    )
    {
        BinEstimator::CoverPair pair_cover = (type == EXCITATORY) ? cache.List()[j].first[i] : cache.List()[j].second[i];
        Number n_bins = system.NumberOfBins();

        if (pair_cover.first._index == pair_cover.second._index) {
            if (pair_cover.first._index >= 0 && pair_cover.first._index < static_cast<int>(n_bins)) {
                Index i_trans = pair_cover.first._index;
                double lower_bound = (1 - pair_cover.first._alpha);
                dydt[system.MapPotentialToProbabilityBin(i)] += rate * (pair_cover.second._alpha - lower_bound) * y[system.MapPotentialToProbabilityBin(i_trans)];
            }
        }

        if (pair_cover.first._index < pair_cover.second._index) {
            if (pair_cover.first._index >= 0 && pair_cover.first._index < static_cast<int>(n_bins))
                dydt[system.MapPotentialToProbabilityBin(i)] += rate * pair_cover.first._alpha * y[system.MapPotentialToProbabilityBin(pair_cover.first._index)];

            if (pair_cover.second._index >= 0 && pair_cover.second._index < static_cast<int>(n_bins))
                dydt[system.MapPotentialToProbabilityBin(i)] += rate * pair_cover.second._alpha * y[system.MapPotentialToProbabilityBin(pair_cover.second._index)];

            for (int j = pair_cover.first._index + 1; j < pair_cover.second._index; j++)
                dydt[system.MapPotentialToProbabilityBin(i)] += rate * y[system.MapPotentialToProbabilityBin(j)];
        }

        assert(!(pair_cover.first._index > pair_cover.second._index));
    }
    void
    AddCachedInputContributionToRow
    (
        double                      dydt[],
        Index                       i,
        const double                y[],
        const AbstractOdeSystem&    system,
        const BinEstimator&         estimator,
        Index                       j,
        const InputParameterSet&    set,
        const CNZLCache&            cache
    )
    {
        // determine the boundaries of this bin; mind the minus sign: exc input comes from lower, inh from higher
        if (set._rate_exc != 0)
            AddCachedToRow(dydt, i, y, system, estimator, j, -set._h_exc, set._rate_exc, cache, EXCITATORY);

        if (set._rate_inh != 0)
            AddCachedToRow(dydt, i, y, system, estimator, j, -set._h_inh, set._rate_inh, cache, INHIBITORY);
    }

    void
    AddCachedRowDeriv
    (
        double                              dydt[],
        Index                               i,
        const double                        y[],
        const AbstractOdeSystem&            system,
        const BinEstimator&                 estimator,
        const vector<InputParameterSet>&    vec_set,
        const CNZLCache&                    cache
    )
    {
        dydt[system.MapPotentialToProbabilityBin(i)] = 0.0;
        Number n_bins  = system.NumberOfBins();
        Number n_input = vec_set.size();
        Index i_reset  = system.IndexResetBin();

        for (Index j = 0; j < n_input; j++) {
            Rate rate_e = vec_set[j]._rate_exc;
            Rate rate_i = vec_set[j]._rate_inh;

            if (rate_i > 0) {
                BinEstimator::CoverPair pair = cache.List()[j].second[0];
                Index i_shift = static_cast<Index>(pair.first._index);
                assert(i_shift >= 0);

                if (i == i_shift)
                    dydt[system.MapPotentialToProbabilityBin(i)] -= (rate_i * pair.first._alpha) * y[system.MapPotentialToProbabilityBin(i)];

                if (i > static_cast<Index>(i_shift))
                    dydt[system.MapPotentialToProbabilityBin(i)] -= rate_i * y[system.MapPotentialToProbabilityBin(i)];
            }


            if (rate_e > 0) {
                if (i == i_reset) {
                	dydt[n_bins] = 0.0;

                    Rate rate = 0;
                    BinEstimator::CoverPair pair = cache.List()[j].first[n_bins - 1];
                    rate += rate_e * (1 - pair.second._alpha) * y[system.MapPotentialToProbabilityBin(pair.second._index)];

                    for (Index ir = pair.second._index + 1; ir < n_bins; ir++)
                        rate += rate_e * y[system.MapPotentialToProbabilityBin(ir)];
                    // n_bins, does exist!
                    dydt[n_bins] += rate;
                }
            }

            dydt[system.MapPotentialToProbabilityBin(i)] -= rate_e * y[system.MapPotentialToProbabilityBin(i)];
            AddCachedInputContributionToRow(dydt, i, y, system, estimator, j, vec_set[j], cache);
        }
    }

    int CachedDeriv(double t, const double y[], double dydt[], void* params)
    {

        MasterParameter* p_par = static_cast<MasterParameter*>(params);
        Number n_bins = p_par->_p_system->NumberOfBins();

        for (Index i = 0; i < n_bins; i++)
            AddCachedRowDeriv(dydt, i, y, *(p_par->_p_system), *(p_par->_p_estimator), *(p_par->_p_vec_set), *(p_par->_p_cache)/*, rate*/);

        return GSL_SUCCESS;
    }
}

NumericalMasterEquation::NumericalMasterEquation
(
    AbstractOdeSystem& 					system,
    const DiffusionParameter&			par_diffusion,
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
        CachedDeriv
    ),
	_queue(TIME_QUEUE_BATCH),
	_rate(0.0),
	_scratch_dense(_system.NumberOfBins()),
	_scratch_pot(_system.NumberOfBins()+1)
{
}

NumericalMasterEquation::~NumericalMasterEquation()
{
}

void NumericalMasterEquation::SortConnectionvector
(
	const std::vector<MPILib::Rate>&									vec_rates,
	const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>&	vec_cons,
	const std::vector<MPILib::NodeType>&								vec_types

)
{
    _convertor.SortConnectionvector(vec_rates,vec_cons,vec_types);
}

void NumericalMasterEquation::InitializeIntegrator()
{
	_integrator.Parameter()._nr_bins    = _system.NumberOfBins();
    _integrator.Parameter()._p_system           = &_system;
    _integrator.Parameter()._p_estimator        = &_estimator;
    _integrator.Parameter()._i_reset            = _system.IndexResetBin();
    _integrator.Parameter()._p_vec_set          = &_convertor.SolverParameter();
    _integrator.Parameter()._p_cache            = &_cache;

}

double NumericalMasterEquation::RecaptureProbability()
{
	// existence of the extra bin is guaranteed in AbstractOdeSytem
	MPILib::Probability p = _system.MassBuffer()[_system.NumberOfBins()];

	_system.MassBuffer()[_system.NumberOfBins()] = 0.0;
	return p;
}

void NumericalMasterEquation::Apply(Time t)
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

Rate NumericalMasterEquation::TransitionRate() const
{
    return  _rate;
}

Rate NumericalMasterEquation::IntegralRate() const
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

	if (i_bound == n_bins - 1){
		return 0.0;
	}

	_scratch_pot[i_bound] = v_bound;
	for (int i = i_bound + 1; i < n_bins; i++)
		_scratch_pot[i] = array_interpretation[i];

	double integ = 0.0;
	_scratch_pot[n_bins] = v_reset;

	for (int i = i_bound; i < n_bins; i++)
		_scratch_dense[i] = array_mass[_system.MapPotentialToProbabilityBin(i)]/(_scratch_pot[i+1]-_scratch_pot[i]);

	for (int i = i_bound; i < n_bins-1; i++)
		integ += 0.5*(_scratch_dense[i]+_scratch_dense[i+1])*(_scratch_pot[i+1]-_scratch_pot[i]);
	integ += 0.5*_scratch_dense[n_bins-1]*(_scratch_pot[n_bins]-_scratch_pot[n_bins-1]);

	Rate rate = _convertor.SolverParameter()[0]._rate_exc;

	return rate*integ;

}
