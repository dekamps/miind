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
#include <boost/lexical_cast.hpp>
#include <MPILib/include/populist/InterpolationRebinner.hpp>
#include <MPILib/include/populist/rateComputation/IntegralRateComputation.hpp>


#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>

namespace MPILib {
namespace populist {
namespace parameters {

PopulistSpecificParameter::PopulistSpecificParameter() :
		_p_rebinner(
				std::shared_ptr<AbstractRebinner>(new InterpolationRebinner)), _p_rate(
				std::shared_ptr<AbstractRateComputation>(new IntegralRateComputation)) {
}

PopulistSpecificParameter::PopulistSpecificParameter(Potential v_min,
		Number n_grid_initial, Number n_add,
		const InitialDensityParameter& par_dens, double fact_expansion,
		const std::string& name_zeroleak, const std::string& name_circulant,
		const std::string& name_noncirculant,
		const AbstractRebinner* p_rebinner,
		const AbstractRateComputation* p_rate) :
		_v_min(v_min), _n_grid_initial(n_grid_initial), _n_add(n_add), _par_dens(
				par_dens), _fact_expansion(fact_expansion), _name_zeroleak(
				name_zeroleak), _name_circulant(name_circulant), _name_noncirculant(
				name_noncirculant), _p_rebinner(
				p_rebinner ?
						std::shared_ptr<AbstractRebinner>(p_rebinner->Clone()) :
						std::shared_ptr<AbstractRebinner>(
								new InterpolationRebinner)), _p_rate(
				p_rate ?
						std::shared_ptr<AbstractRateComputation>(
								p_rate->Clone()) :
						std::shared_ptr<IntegralRateComputation>(
								new IntegralRateComputation)) {
}

PopulistSpecificParameter::PopulistSpecificParameter(
		const PopulistSpecificParameter& rhs) :
		_v_min(rhs._v_min), _n_grid_initial(rhs._n_grid_initial), _n_add(
				rhs._n_add), _par_dens(rhs._par_dens), _fact_expansion(
				rhs._fact_expansion), _name_zeroleak(rhs._name_zeroleak), _name_circulant(
				rhs._name_circulant), _name_noncirculant(
				rhs._name_noncirculant), _p_rebinner(
				(rhs._p_rebinner.get() == nullptr) ?
				rhs._p_rebinner :
				std::shared_ptr<AbstractRebinner>(
						rhs._p_rebinner->Clone())), _p_rate(
						(rhs._p_rate.get() == nullptr) ?
						rhs._p_rate :
						std::shared_ptr<AbstractRateComputation>(
								rhs._p_rate->Clone())) {
				}

PopulistSpecificParameter::~PopulistSpecificParameter() {
}

PopulistSpecificParameter& PopulistSpecificParameter::operator=(
		const PopulistSpecificParameter& rhs) {
	if (this == &rhs)
		return *this;

	_v_min = rhs._v_min;
	_n_grid_initial = rhs._n_grid_initial;
	_n_add = rhs._n_add;
	_par_dens = rhs._par_dens;
	_fact_expansion = rhs._fact_expansion;
	_p_rebinner = (rhs._p_rebinner.get() == nullptr) ?
	rhs._p_rebinner :
	std::shared_ptr<AbstractRebinner>(rhs._p_rebinner->Clone());
	_p_rate = (rhs._p_rate.get() == nullptr) ?
	rhs._p_rate :
	std::shared_ptr<AbstractRateComputation>(
			rhs._p_rate->Clone());
	_name_zeroleak = rhs._name_zeroleak;
	_name_circulant = rhs._name_circulant;
	_name_noncirculant = rhs._name_noncirculant;

	return *this;
}

Potential PopulistSpecificParameter::getVMin() const {
	return _v_min;
}

Number PopulistSpecificParameter::getNrGridInitial() const {
	return _n_grid_initial;
}

InitialDensityParameter PopulistSpecificParameter::getInitialDensity() const {
	return _par_dens;
}

PopulistSpecificParameter* PopulistSpecificParameter::Clone() const {
	return new PopulistSpecificParameter(*this);
}

Number PopulistSpecificParameter::getMaxNumGridPoints() const {
	return static_cast<Number>(_n_grid_initial * _fact_expansion);
}

const AbstractRebinner& PopulistSpecificParameter::getRebin() const {
	return *_p_rebinner;
}

const AbstractRateComputation& PopulistSpecificParameter::getRateComputation() const {
	return *_p_rate;
}

double PopulistSpecificParameter::getExpansionFactor() const {
	return _fact_expansion;
}

Number PopulistSpecificParameter::getNrAdd() const {
	return _n_add;
}

std::string PopulistSpecificParameter::getZeroLeakName() const {
	return _name_zeroleak;
}

std::string PopulistSpecificParameter::getCirculantName() const {
	return _name_circulant;
}

std::string PopulistSpecificParameter::getNonCirculantName() const {
	return _name_noncirculant;
}
} //end namespace parameters
} //end namespace populist
} //end namespace MPILib
