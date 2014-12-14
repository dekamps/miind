// Copyright (c) 2005 - 2014 Marc de Kamps
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

#ifndef _CODE_LIBS_GEOMLIB_GEOMINPUTCONVERTOR_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_GEOMINPUTCONVERTOR_INCLUDE_GUARD

#include <vector>
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>
#include <MPILib/include/NodeType.hpp>
#include "CurrentCompensationParameter.hpp"
#include "DiffusionParameter.hpp"
#include "InputParameterSet.hpp"
#include "MuSigmaScalarProduct.hpp"
#include "OrnsteinUhlenbeckParameter.hpp"


namespace GeomLib {


	class GeomInputConvertor {
	public:

		GeomInputConvertor
		(
			const OrnsteinUhlenbeckParameter&,
			const DiffusionParameter&,
			const CurrentCompensationParameter&,
			const std::vector<MPILib::Potential>&,
			bool  force_small_bins = false
		);


		void SortConnectionvector
		(
			const std::vector<MPILib::Rate>&,
			const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>&,
			const std::vector<MPILib::NodeType>&
		);

		std::vector<InputParameterSet>& SolverParameter() { assert (_vec_set.size() > 0 ); return  _vec_set; }

		const std::vector<InputParameterSet>& SolverParameter() const { assert(_vec_set.size() > 0); return _vec_set; }

		//! number of external inputs that are interpreted as direct input
		MPILib::Number NumberDirect() const;

	private:

		void AddDiffusionParameter
		(
			const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>&,
			const std::vector<MPILib::Rate>&
		);

		void AddBurstParameters
		(
			const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>&,
			const std::vector<MPILib::Rate>&
		);

		void SetDiffusionParameters
		(
			const MuSigma& par,
			InputParameterSet& set
		) const;


		void SortDiffusionInput
		(
			const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>&,
			const std::vector<MPILib::Rate>& vec_rates,
			std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>*,
			std::vector<MPILib::Rate>*

		);
		bool IsSingleDiffusionProcess(MPILib::Potential h) const;

		MPILib::Potential MinVal(const std::vector<MPILib::Potential>&) const;

		const OrnsteinUhlenbeckParameter		_par_neuron;
		const DiffusionParameter				_par_diff;
		const CurrentCompensationParameter		_par_curr;
		std::vector<MPILib::Potential>			_vec_interpretation;
		std::vector<InputParameterSet>			_vec_set;

		std::vector<MPILib::Index>				_vec_direct;
		std::vector<MPILib::Index>				_vec_diffusion;

		MPILib::Potential						_min_step;
/*		double						_gamma_estimated;
		double						_minval;
		MPILib::Potential					_V_min;
		MPILib::Potential					_V_max;
		MPILib::Time						_tau;
		MPILib::Number						_n_bins;
		MPILib::Index*	 		_p_index_reversal_bin;
		bool						_b_toggle_sort;
		bool						_b_toggle_diffusion;
		double						_diffusion_limit;
		double						_diffusion_step;
		double						_h_burst_min;
		double						_dc_component;
		double						_sigma_fraction;
		bool						_no_sigma_smooth;*/
		bool						_force_small_bins;


	};
}

#endif //include guard
