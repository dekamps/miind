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
#include <GeomLib/AbstractOdeSystem.hpp>
#include "GeomLibException.hpp"

using namespace GeomLib;

AbstractOdeSystem::AbstractOdeSystem
(
 const AbstractNeuralDynamics& dyn
):
_p_dyn					(boost::shared_ptr<AbstractNeuralDynamics>(dyn.Clone())),
_t_period				(_p_dyn->TPeriod()),
_t_step					(_p_dyn->TStep()),
_par					(_p_dyn->Par()),
_buffer_interpretation	(dyn.InterpretationArray()),
_buffer_mass	       	(this->InitializeDensity()),
_i_reset 	       		(this->InitializeResetBin()),
_i_reversal				(this->InitializeReversalBin()),
_t_current   	       	(0.0),
_map_cache				(vector<Index>(0)),
_number_of_bins			(_buffer_interpretation.size())
{
}

AbstractOdeSystem::AbstractOdeSystem
(
 const AbstractOdeSystem& sys
):
_p_dyn					(boost::shared_ptr<AbstractNeuralDynamics>(sys._p_dyn->Clone())),
_t_period				(sys._t_period),
_t_step					(sys._t_step),
_par					(sys._par),
_buffer_interpretation	(sys._buffer_interpretation),
_buffer_mass	       	(sys._buffer_mass),
_i_reset      			(sys._i_reset),
_i_reversal				(sys._i_reversal),
_t_current     			(sys._t_current),
_map_cache				(sys._map_cache),
_number_of_bins			(sys._number_of_bins)
{
}

AbstractOdeSystem::~AbstractOdeSystem()
{
}

Index AbstractOdeSystem::InitializeResetBin() const
{
	Index i_ret;
	i_ret = this->FindBin(_par._par_pop._V_reset);
	return i_ret;
}


Index AbstractOdeSystem::InitializeReversalBin() const
{
	Index i_ret;
	i_ret = this->FindBin(_par._par_pop._V_reversal);
	return i_ret;
}

Index AbstractOdeSystem::FindBin(Potential V) const
{
	Index i_reset_bin;
	if ( V < _par._V_min || V > _par._par_pop._theta)
		throw GeomLibException("Reset potential doesn't make sense");

	if ( V > _buffer_interpretation.back()){
		i_reset_bin = _buffer_interpretation.size() - 1;
		return i_reset_bin;
	} else
		for (Index i = 0; i < _buffer_interpretation.size() - 1; i++ ){
			if ( V >= _buffer_interpretation[i] && V < _buffer_interpretation[i+1] ){
				i_reset_bin = i;
				return i_reset_bin;
			}
		}
	throw GeomLibException("LeakingOdeSystem couldn't resolve reset bin");
}

void AbstractOdeSystem::PrepareReport
(
 double* array_interpretation,
 double* array_mass
) const
{
  Number n_bins = this->NumberOfBins();

  for (Index i = 0; i < n_bins-1; i++){
    array_interpretation[i] = _buffer_interpretation[i];
    array_mass[i] = _buffer_mass[ MapPotentialToProbabilityBin(i)]/(_buffer_interpretation[i+1]-_buffer_interpretation[i]);
  }

  array_interpretation[n_bins-1] = _buffer_interpretation[n_bins-1];
  array_mass[n_bins-1] = _buffer_mass[MapPotentialToProbabilityBin(n_bins-1)]/(this->_par._par_pop._theta - _buffer_interpretation[n_bins-1]);
}

void AbstractOdeSystem::InitializeSingleBin(vector<MPILib::Density>* p_vec) const
{
	vector<MPILib::Density>& buffer_mass = *p_vec;
	const OdeParameter& par_ode = this->Par();
	Number n_bins = p_vec->size();

	assert (par_ode._par_dens._mu >= this->_par._V_min);
	assert (par_ode._par_dens._mu <= this->_par._par_pop._theta);

	Index j = this->FindBin(par_ode._par_dens._mu);
	if ( j == n_bins - 1 )
		buffer_mass[j] = 1.0/(this->_par._par_pop._theta - _buffer_interpretation[n_bins -1]);
	else
		buffer_mass[j] = 1.0/(_buffer_interpretation[j+1] - _buffer_interpretation[j]);
}


void AbstractOdeSystem::InitializeGaussian(vector<MPILib::Density>* p_vec_dense) const
{
	vector<MPILib::Density>& buffer_mass = *p_vec_dense;
	const OdeParameter& par_ode = this->Par();
	Number n_bins = p_vec_dense->size();
	for (Index i = 0; i < n_bins; i++)
	{
		double v   = _buffer_interpretation[i];
		double sqr = (v - par_ode._par_dens._mu)*(v - par_ode._par_dens._mu)/(par_ode._par_dens._sigma*par_ode._par_dens._sigma);
		buffer_mass[i] = exp(-sqr);
	}
}

vector<MPILib::Density> AbstractOdeSystem::InitializeDensity() const
{
	vector<MPILib::Density> vec_dense(_buffer_interpretation.size()+1); // TODO: review, rather hacky
	if (this->Par()._par_dens._sigma == 0.0)
		this->InitializeSingleBin(&vec_dense);
	else
		this->InitializeGaussian(&vec_dense);

	this->NormaliseDensity(&vec_dense);
	return vec_dense;
}

void AbstractOdeSystem::NormaliseDensity
(
	vector<MPILib::Density>* p_vec
) const
{
	vector<MPILib::Density>& buffer_mass = *p_vec;
	double sum = 0;
	Number n_bins = p_vec->size();

	for (Index i = 0; i < n_bins - 1; i++)
		buffer_mass[i] *= (_buffer_interpretation[i+1] - _buffer_interpretation[i]);

	buffer_mass[n_bins-1] *= (this->_par._par_pop._theta - _buffer_interpretation[n_bins-1]);

	for (Index i = 0; i < n_bins; i++)
		sum += buffer_mass[i];

	for (Index i = 0; i < n_bins; i++)
		buffer_mass[i] /= sum;
}


