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

#ifndef MPILIB_POPULIST_ZEROLEAKEQUATIONS_INPUTCONVERTOR_HPP_
#define MPILIB_POPULIST_ZEROLEAKEQUATIONS_INPUTCONVERTOR_HPP_

#include <vector>

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>

#include <MPILib/include/DelayedConnection.hpp>
#include <MPILib/include/populist/parameters/InputParameterSet.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/zeroLeakEquations/MuSigma.hpp>
#include <MPILib/include/populist/zeroLeakEquations/MuSigmaScalarProduct.hpp>
#include <MPILib/include/populist/zeroLeakEquations/SpecialBins.hpp>
#include <MPILib/include/BasicDefinitions.hpp>


namespace MPILib {
namespace populist {
//forward declarations
  namespace parameters {
	class PopulistSpecificParameter;
  }
	//! Converts external input from other populations in parameters that are suitable for
	//! AbstractCirculantSolver and AbstractNonCirculantSolver objects.

	//! PopulationAlgorithm receives the contribution from its input populations as a list of pointer weight pairs. Via the
	//! pointers, the input nodes can be queried for their current activation, via the weights their contribution be, well, weighted.
	//! The list also contains hints as to whether for some inputs they should be interpreted as Gaussian white noise, in which case
	//! they can be combined into a single input contribution in a way explaned below. If they cannot be combined in such a way,
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

	class InputConvertor {
	public:

		//! Constructor registers where the output results must be written, which can happen at construction of PopulationGridController.
		//! Note that constructor grants access to variables that are maintained elsewhere, in PopulationGridController and that there
		//! may be some redundanccy in the information available at this level.
		InputConvertor
		(
		        zeroLeakEquations::SpecialBins&				bins,
       			parameters::PopulationParameter&		par_pop,
       			parameters::PopulistSpecificParameter&	par_spec,
			Potential&					delta_v,
			Number&						n_bins,
			double						diffusion_limit = 0.05,
			double						diffusion_step  = 0.03

		):
		_p_bins(&bins),
		_vec_set(1),		// many older algorithms, OldLIFZeroLeakEquations, SingleInputZeroLeakEquations simply assume there is one InputSetParameter
		_p_par_pop(&par_pop),
		_p_par_spec(&par_spec),
		_p_delta_v(&delta_v),
		_p_n_bins(&n_bins),
		_b_toggle_sort(false),
		_b_toggle_diffusion(false),
		_diffusion_limit(diffusion_limit),
		_diffusion_step(diffusion_step)
		{
		}

		//! A signaller for when the PopulationGridController starts to configure
		void Configure
		(
		     std::valarray<Potential>&
		);


	/**
	 * This function collects the external input, and lays it out internally for use in AdaptParameters.
	 * This function must be called when input charateristics change, for example at the start of an Evolve.
	 * It used to be part of AdaptParameters, but has been moved out of there to help account for synchronous
	 * updating in networks. MPINetwork will now first go through a loop calling nodes to update their
	 * input, and then a second loop to evolve their states.
	 * @param nodeVector The vector which stores the Rates of the precursor nodes
	 * @param weightVector The vector which stores the Weights of the precursor nodes
	 * @param typeVector The vector which stores the NodeTypes of the precursor nodes
	 */
	void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<DelayedConnection>& weightVector,
			const std::vector<NodeType>& typeVector);


		//! This function interprets the  input contribution from other populations and converts them into input parameters for the
		//! circulant and non circulant solvers. It must be run every time _delta_v changes.
		void AdaptParameters();

		//! This function assumes that the current input is known and stored in the input parameter set, but that the parameters
		//! for the circulant and non-circulant solver is wrong. One case where this happens is after rebinning: the input firing rates
		//! and efficacies have changed, but the solver parameters are invalid. This function recalculates them.
		void RecalculateSolverParameters
		(
		);

	  std::vector<parameters::InputParameterSet>& getSolverParameter(){ return  _vec_set; }

	  const std::vector<parameters::InputParameterSet>& SolverParameter() const { return _vec_set; }


	/**
	 * Const Getter for the PopulationParameter
	 * @return A const reference to the PopulationParameter
	 */

	  const parameters::PopulationParameter& getParPop() const { return *_p_par_pop; }


	/**
	 * Getter for the reversal Bins
	 * @return A reference to the reversal bins
	 */
	const Index& getIndexReversalBin() const;


	/**
	 * Getter for the reversal Bins
	 * @return A reference to the reversal bins
	 */
	const Index& getIndexCurrentResetBin() const;
	/**
	 * Purpose: after someone has changed _p_input_set->_H_exc, ..inh, the number
	 * of non_circulant bins must be adapted
	 */
        void UpdateRestInputParameters();

	//! Check whether or not two inputs are required to emulate diffusion
	double DiffusionLimit () const;

        //! Fraction of the difference between threshold and reversal potential that is used to emulate
	//! diffusion.
	double DiffusionStep() const {return _diffusion_step; }


	//! Potential jump in membrane potential units (e.g. V) used to emulate diffusion.
	double DiffusionJump() const { return _diffusion_step*(_p_par_pop->_theta - _p_par_pop->_V_reversal); }

	void SetDiffusionParameters
	(
	 const zeroLeakEquations::MuSigma&,
	      parameters::InputParameterSet&
	) const;

	private:

		void AddDiffusionParameter
		(
		        const std::vector<Rate>& nodeVector,
			const std::vector<DelayedConnection>& weightVector
		);

		void AddBurstParameters		
		(
		        const std::vector<Rate>& nodeVector,
			const std::vector<DelayedConnection>& weightVector
                );

		bool IsSingleDiffusionProcess(Potential h) const;


		Time*						_p_time;
	        zeroLeakEquations::SpecialBins*			_p_bins;
	        std::vector<parameters::InputParameterSet>	        _vec_set;
	        parameters::PopulationParameter*		_p_par_pop;
	        parameters::PopulistSpecificParameter*	        _p_par_spec;
		Rate**						_pp_rate;
		Potential*					_p_delta_v;
		Number*						_p_n_bins;
		Index*						_p_index_reversal_bin;
		bool						_b_toggle_sort;
		bool						_b_toggle_diffusion;
		double						_diffusion_limit;
		double						_diffusion_step;
	
	  std::vector<Index>			    _vec_burst;
	  std::vector<Index>			    _vec_diffusion;
	  std::vector<DelayedConnection>  _vec_diffusion_weight;
	};
}
}
#endif //include guard

