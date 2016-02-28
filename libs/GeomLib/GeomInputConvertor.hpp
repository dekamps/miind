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
#include <MPILib/include/DelayedConnection.hpp>
#include <MPILib/include/NodeType.hpp>
#include "CurrentCompensationParameter.hpp"
#include "DiffusionParameter.hpp"
#include "InputParameterSet.hpp"
#include "MuSigmaScalarProduct.hpp"
#include "NeuronParameter.hpp"


namespace GeomLib {

	//! Internally used by GeomLib, interprets input from white noise; calculates the current compensation contribution.

	//! This object is handed a list of firing rates, node types and efficacies and converts this
	//! to another list of firing rates that is used internally by GeomAlgorithm. Some inputs to a population
	//! are to be interpreted as contributions to Gaussian white noise. Others should simply be passed
	//! on to GeomAlgorithm. Internally, GeomAlgorithm uses a single input to emulate Gaussian white noise,
	//! which means that GeomInputConvertor must collapse the external white noise contribution into a single
	//! one that can be used internally.
	class GeomInputConvertor {
	public:

		GeomInputConvertor
		(
			const NeuronParameter&,	       			//! Neuron parameter of the receiving population, required to be able to interpret the white noise contribution
			const DiffusionParameter&,     			//! Determines when white noise is emulated internally by one or two Poisson inputs
			const CurrentCompensationParameter&,   	//! Creates an extra internal white noise source that implements the current compensation
			const std::vector<MPILib::Potential>&, 	//! Interpretation array from the relevant AbstractOdeSystem
			bool  force_small_bins = false 			//! Inactive at the moment
		);


		void SortConnectionvector
		(
			const std::vector<MPILib::Rate>&,
			const std::vector<MPILib::DelayedConnection>&,
			const std::vector<MPILib::NodeType>&
		);

		std::vector<InputParameterSet>& SolverParameter() { assert (_vec_set.size() > 0 ); return  _vec_set; }

		const std::vector<InputParameterSet>& SolverParameter() const { assert(_vec_set.size() > 0); return _vec_set; }

		//! number of external inputs that are interpreted as direct input
		MPILib::Number NumberDirect() const;

	private:

		void AddDiffusionParameter
		(
			const std::vector<MPILib::DelayedConnection>&,
			const std::vector<MPILib::Rate>&
		);

		void AddBurstParameters
		(
			const std::vector<MPILib::DelayedConnection>&,
			const std::vector<MPILib::Rate>&
		);

		void SetDiffusionParameters
		(
			const MuSigma& par,
			InputParameterSet& set
		) const;


		void SortDiffusionInput
		(
			const std::vector<MPILib::DelayedConnection>&,
			const std::vector<MPILib::Rate>& vec_rates,
			std::vector<MPILib::DelayedConnection>*,
			std::vector<MPILib::Rate>*

		);

		bool IsSingleDiffusionProcess(MPILib::Potential h) const;

		MPILib::Potential MinVal(const std::vector<MPILib::Potential>&) const;

		const NeuronParameter	            	_par_neuron;
		const DiffusionParameter      			_par_diff;
		const CurrentCompensationParameter     	_par_curr;
		std::vector<MPILib::Potential>	       	_vec_interpretation;
		std::vector<InputParameterSet>	       	_vec_set;

		std::vector<MPILib::Index>    		_vec_direct;
		std::vector<MPILib::Index>     		_vec_diffusion;

		MPILib::Potential    		       	_min_step;

		bool				      	_force_small_bins;


	};
}

#endif //include guard
