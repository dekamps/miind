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

//#include "../DynamicLib/DynamicLib.h"
#include "InputParameterSet.hpp"
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>
#include "MuSigmaScalarProduct.hpp"
#include "OrnsteinUhlenbeckParameter.hpp"


//using DynamicLib::AbstractAlgorithm;
//using DynamicLib::DynamicNode;
//using DynamicLib::ReportValue;

namespace GeomLib {

	class PopulistSpecificParameter;
	struct SpecialBins;
	
	//! Converts external input from other populations in parameters that are suitable for
	//! AbstractCirculantSolver and AbstractNonCirculantSolver objects.

	//! PopulationAlgorithm receives the contribution from its input populations as a list of pointer weight pairs. Via the
	//! pointers, the input nodes can be queried for their current activation, via the weights their contribution be, well, weighted.
	//! The list also contains hints as to whether for some inputs they should be interpreted as Gaussian white noise, in which case
	//! they can be combined into a single input contribution in a way explained below. If they cannot be combined in such a way,
	//! input parameters are calculated for each input population individually.
	//! The ZeroLeakEquation solvers operate on an object of type InputParameterSet. InputConvertor maintains a vector of these
	//! objects, which is accessible by the SolverParameter method. By convention, the first element of this vector, indexed by 0 are 
	//! those inputs that can be lumped together as
	//! 'white noise', i.e. their input will be aggregated into a mean and variance from which an effective stepsize and rate
	//! are calculated, or effective stepsizes and rates if tow input populations are required to represent that white noise,
	//! as is the case for balanced excitation-inhibition. 
	//!
	//! There is always at least one element in the vector of InputParameterset objects that can be accessed using the SolverParameter
	//! method. If there is input that can be considered to be Gausssian white noise, it will be represented by the first element.

	class CharacteristicInputConvertor {
	public:

//		typedef AbstractSparseNode<double,OrnsteinUhlenbeckConnection>::predecessor_iterator predecessor_iterator;

//		typedef MuSigma ScalarProductParameterType;

		//! Constructor registers where the output results must be written, which can happen at construction of PopulationGridController.
		//! Note that constructor grants access to variables that are maintained elsewhere, in PopulationGridController and that there
		//! may be some redundancy in the information available at this level.
		CharacteristicInputConvertor
		(
			MPILib::Time						tau,
			const std::vector<MPILib::Potential>&	vec_interpretation,
			const std::vector<MPILib::Density>&		vec_density,
			MPILib::Potential					dc_component     = 0.0,
			double						diffusion_limit  = 0.05,
			double						diffusion_step   = 0.03,
			double						sigma_smooth     = 0.01,  //! the dc component that is subtracted from the input to compensate for using a positive input current must have a small variability
			bool						no_sigma_smooth  = false, //! set to true if you want to deactivate the sigma correction (in general you should not do this, except for testing)
			bool						force_small_bins = false  //! set to true if you want to work without limits on the minimum steps size (this is not recommened but can be useful for testing)
		);


		//! A signaller for when the PopulationGridController starts to configure
		void Configure
		(
			std::valarray<MPILib::Potential>&
		);

		//! This function collects the external input, and lays it out internally for use in AdaptParameters. This function must
		//! be called when input charateristics change, for example at the start of an Evolve. It used to be part of AdaptParameters,
		//! but has been moved out of there to help account for synchronous updating in networks. DynamicNetworkImplementation will now
		//! first go through a loop calling nodes to update their input, and then a second loop to evolve their states.
		void SortConnectionvector
		(
			const std::vector<MPILib::Rate>&,
			const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>&,
			const std::vector<MPILib::NodeType>&
		);
/*
		//! This function interprets the  input contribution from other populations and converts them into input parameters for the
		//! circulant and non circulant solvers. It must be run every time _delta_v changes.
		void AdaptParameters();
*/
/*		//! This function assumes that the current input is known and stored in the input parameter set, but that the parameters
		//! for the circulant and non-circulant solver is wrong. One case where this happens is after rebinning: the input firing rates
		//! and efficacies have changed, but the solver parameters are invalid. This function recalculates them.
		void RecalculateSolverParameters
		(
		);
*/
		std::vector<InputParameterSet>& SolverParameter(){ return  _vec_set; }

		const std::vector<InputParameterSet>& SolverParameter() const { return _vec_set; }

		void UpdateRestInputParameters();

		//! Check whether or not two inputs are required to emulate diffusion
		double DiffusionLimit () const;

		//! Fraction of the difference between threshold and reversal potential that is used to emulate
		//! diffusion.
		double DiffusionStep() const {return _diffusion_step; }


		//! Potential jump in membrane potential units (e.g. V) used to emulate diffusion.
		double DiffusionJump() const { return _diffusion_step*(_V_max - _V_min); }

		void SetDiffusionParameters
		(
			const MuSigma&,
			InputParameterSet&
		) const;

		double EstimatedGamma() const{ return _gamma_estimated; }

#ifdef _INVESTIGATE_ALGORITHM
		vector<ReportValue>& Values(){ return _vec_value; }
#endif
	private:

		MPILib::Potential V_min_from_gamma(MPILib::Potential) const;
		MPILib::Potential V_max_from_gamma(MPILib::Potential) const;

//		typedef pair<AbstractSparseNode<double,OrnsteinUhlenbeckConnection>*,OrnsteinUhlenbeckConnection> connection;

		void AddDiffusionParameter	();
		void AddBurstParameters		();

		void AdaptSigma(MPILib::Potential, MPILib::Potential*, MPILib::Potential*) const;

		bool IsSingleDiffusionProcess(MPILib::Potential h) const;


//		predecessor_iterator DCComponent(double);

		double EstimateGamma(const std::vector<double>&) const;
		double MinVal(const std::vector<MPILib::Potential>&) const;

		std::vector<MPILib::Potential>			_vec_interpretation;
		std::vector<InputParameterSet>	_vec_set;
		double						_gamma_estimated;
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
		bool						_no_sigma_smooth;
		bool						_force_small_bins;


//		DynamicNode<OU_Connection>  _node;

//		connection								_connection;

//		vector<predecessor_iterator>			_vec_burst;
//		vector<predecessor_iterator>			_vec_diffusion;
	};
}

#endif //include guard
