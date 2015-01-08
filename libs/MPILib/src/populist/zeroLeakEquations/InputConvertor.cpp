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

#include <iostream>
#include <cassert>
#include <boost/foreach.hpp>
#include <MPILib/include/populist/zeroLeakEquations/InputConvertor.hpp>
using namespace MPILib;
using namespace populist;
using namespace zeroLeakEquations;

void InputConvertor::AdaptParameters
(
)
{
	// Purpose: Adaption from stride size to current scale, to be called after every new AddBin, or rebinning
	// Assumes correct values for _delta_v and _set_input, so DeltaV() must be called before this routine
	// Author: Marc de Kamps
	// Date: 21-09-2005

	// Additions: 20-10-2005 One and Two Population modes
	//          : 01-01-2006 separate functions for One and Two Population modes
	//			: 23-03-2009 Move from PopulationGridController to ConvertMuSigmaToH
	//			: 28-07-2011 Now directly responsible for reading the input contribution of other populations. This version now can not interpret diffusion input anymore.

	RecalculateSolverParameters();
	UpdateRestInputParameters();
}


void InputConvertor::SetDiffusionParameters
(
	const MuSigma& par,
	parameters::InputParameterSet& set
) const
{
	double mu    = par._mu;
	double sigma = par._sigma;
	

	if (mu == 0.0 && sigma ==0.0){
		set._h_exc = 0.0;
		set._h_inh = 0.0;
		set._rate_exc = 0.0;
		set._rate_inh = 0.0;
		return;
	}
	Potential h = (mu != 0.0) ? sigma*sigma/mu : std::numeric_limits<double>::max();

	if (mu != 0 && IsSingleDiffusionProcess(h) ){
		Rate rate = mu*mu/(sigma*sigma*_p_par_pop->_tau);
		if ( h > 0 ){
			set._h_exc    = h;
			set._h_inh    = 0.0;
			set._rate_exc = rate;
			set._rate_inh = 0.0;

		}
		else {
			set._h_exc	= 0.0;
			set._h_inh	=  h; // removed - sign, MdK 04/08/2012; the step size need to be forwarded as a negative number
			set._rate_exc = 0.0;
			set._rate_inh = rate;
		}
	}
	else {
		double h = this->DiffusionJump();
		double tau = _p_par_pop->_tau;
		set._h_exc = h;
		set._h_inh = -h;

		set._rate_exc = (sigma*sigma + h*mu)/(2*h*h*tau);
		set._rate_inh = (sigma*sigma - h*mu)/(2*h*h*tau);
	}
}

bool InputConvertor::IsSingleDiffusionProcess(Potential h) const
{
	return (fabs(h/(_p_par_pop->_theta - _p_par_pop->_V_reversal)) < _diffusion_limit);
}

void InputConvertor::UpdateRestInputParameters
(
)
// Purpose: after someone has changed _p_input_set->_H_exc, ..inh, the number
// of non_circulant bins must be adapted
// Author: M. de Kamps
// Date: 26-06-2008
// Modification: 23-03-2009; Moved from PopulationGridController to ConvertMuSigmaToH
{
  BOOST_FOREACH(parameters::InputParameterSet& set, _vec_set){
		int remainder              = (set._H_exc != 0 && (*_p_n_bins)%set._H_exc == 0 ) ? 0 : 1;
		set._n_noncirc_exc  = (set._H_exc != 0) ? (*_p_n_bins)/set._H_exc + remainder : 0;

		remainder                  = (set._H_inh != 0 && (*_p_n_bins)%set._H_inh == 0 ) ? 0 : 1;
		set._n_noncirc_inh  = (set._H_inh != 0) ? (*_p_n_bins)/set._H_inh + remainder : 0;

		set._n_circ_exc     = (set._H_exc != 0) ? 
									 static_cast<Number>((_p_par_pop->_theta - _p_par_pop->_V_reset)/set._h_exc) + 1 : 
									 set._n_circ_exc = 0;
	}
}

void InputConvertor::RecalculateSolverParameters
( 
)
{
	// _delta_v != 0 is guaranteed
	// This step can NOT be moved into UpdateRestParameters, because some versions
	// of the algorithm mess with the values computed here and then call Update....

  BOOST_FOREACH(parameters::InputParameterSet& set, _vec_set){
		set._H_exc = static_cast<int>(floor(set._h_exc/(*_p_delta_v)));
		set._H_inh = static_cast<int>(floor(-set._h_inh/(*_p_delta_v)));

		assert(set._H_exc >= 0);
		assert(set._H_inh >= 0);

		set._alpha_exc = set._h_exc/(*_p_delta_v) - set._H_exc;
		set._alpha_inh = -set._h_inh/(*_p_delta_v) - set._H_inh;
	}
}

void InputConvertor::Configure
(
 std::valarray<Potential>& array_state
)
{
}

const Index& InputConvertor::getIndexReversalBin() const
{
	return _p_bins->_index_reversal_bin;
}

const Index& InputConvertor::getIndexCurrentResetBin() const
{
	return _p_bins->_index_current_reset_bin;
}

void InputConvertor::SortConnectionvector
(
                        const std::vector<Rate>& nodeVector,
			const std::vector<DelayedConnection>& weightVector,
			const std::vector<NodeType>& typeVector	
)
{

         assert ( nodeVector.size() == weightVector.size());
         assert ( nodeVector.size() == typeVector.size() );

	Number n_inputs = nodeVector.size();
	assert (n_inputs != 0 );

	// sorting depends on network structure and only should be done once
	if (! _b_toggle_sort ) {
	  for ( Index i = 0; i < n_inputs; i++){

	    if ( typeVector[i] == EXCITATORY_DIRECT || typeVector[i] == INHIBITORY_DIRECT ){
	      _vec_burst.push_back(i);
	    }
	    else {
	      _vec_diffusion.push_back(i);
	    }
	  }
	  _b_toggle_sort = true;
	}

	for (Index i = 0; i < _vec_diffusion.size(); i++)
	  _vec_diffusion_weight.push_back(weightVector[_vec_diffusion[i]]);

	this->AddDiffusionParameter(nodeVector,weightVector);
	this->AddBurstParameters(nodeVector,weightVector);
  
}

void InputConvertor::AddBurstParameters
(
    const std::vector<Rate>& nodeVector,
    const std::vector<DelayedConnection>& weightVector
)
{
	// This is necessary for older ZeroLeakEquations. They rely on the relevant input 
	// being in _vec-set[0]. So if there is no diffusion input, the first element has
	// to be taken up by bursting input.

	if (_vec_diffusion.size() == 0 )
		_vec_set.clear();

	  for (Index ind = 0; ind < _vec_burst.size(); ind++){
	        double h = weightVector[_vec_burst[ind]]._efficacy;
		double N = weightVector[_vec_burst[ind]]._number_of_connections;
		double rate = nodeVector[_vec_burst[ind]];

		parameters::InputParameterSet set;
		if (h >= 0 ){
			set._h_exc    = h;
			set._h_inh    = 0;
			set._rate_exc = rate*N;
			set._rate_inh = 0.0;
		} else
		{
			set._h_exc	 = 0;
			set._h_inh	 = h;
			set._rate_exc = 0.0;
			set._rate_inh = rate*N;
		}

		_vec_set.push_back(set);
	}
}

void InputConvertor::AddDiffusionParameter
(
	const std::vector<Rate>& nodeVector,
	const std::vector<DelayedConnection>& weightVector
)
{	
        std::vector<Rate> vec_diffusion_rate;
        for ( Index i = 0; i < _vec_diffusion.size(); i++ ){
	     vec_diffusion_rate.push_back(nodeVector[_vec_diffusion[i]]);
	     assert( vec_diffusion_rate[i] >= 0.0);
	}

        zeroLeakEquations::MuSigmaScalarProduct scalar;
	MuSigma par = 
		scalar.Evaluate
		(
			vec_diffusion_rate,
			_vec_diffusion_weight,
			_p_par_pop->_tau
		);

	SetDiffusionParameters(par,_vec_set[0]);
}
