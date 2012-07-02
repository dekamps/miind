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
#ifndef MPILIB_POPULIST_ABSTRACTZEROLEAKEQUATIONS_HPP_
#define MPILIB_POPULIST_ABSTRACTZEROLEAKEQUATIONS_HPP_

#include "SpecialBins.h"

#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>
#include <MPILib/include/populist/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/populist/AbstractCirculantSolver.hpp>
#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/InputParameterSet.hpp>

namespace MPILib {
namespace populist {


	//! A solver for the zero leak master equations in the PopulationAlgorithm.
	
	//! PopulationAlgorithm models the combined effect from leaky-integrate-and-fire (LIF) dynamics and Poisson input
	//! spike trains on individual neurons. The effects of LIF dynamics are accounted for by maintaining the density
	//! in a PopulationGridController. The PopulationGridController implements exponential shrinkage (LIF decay) by
	//! relabeling the potential density every time step and by adding points. The principle is explained in \ref population_algorithm.
	//! Every time step the M equation governing the Poisson statistics of input spike trains must be executed.
	//! This is handled by ZeroLeakEquations.
	class AbstractZeroLeakEquations {
	public:

		//! Constructor, giving access to most relevant state variables held by PopulationGridController
		AbstractZeroLeakEquations
		(
			VALUE_REF_INIT
			Number&,										//!< reference to the current number of bins
			valarray<Potential>&			state,			//!< reference to state array
			Potential&,										//!< reference to the check sum variable
			SpecialBins&					bins,		
			PopulationParameter&			par_pop,		//!< reference to the PopulationParameter 
			PopulistSpecificParameter&		par_spec,		//!< reference to the PopulistSpecificParameter
			Potential&						delta_v			//!< reference to the current scale variable
		):_array_state(state),_par_pop(par_pop),_par_spec(par_spec),_bins(bins),_p_set(0){}

		virtual ~AbstractZeroLeakEquations(){};


		//! Pass in whatever other parameters are needed. This is explicitly necessary for OneDMZeroLeakEquations
		virtual void Configure
		(
			void*
		) = 0;

		//! Given input parameters, derived classes are free to implement their own solution for ZeroLeakEquations
		virtual void Apply(Time) = 0;

		//! Every Evolve step (but not every time step, see below), the input parameters must be updated
		virtual void SortConnectionvector
		(
			predecessor_iterator,
			predecessor_iterator
		) = 0;

		//! Every time step the input parameters must be adapated, even if the input doesn't change, because the are affected
		//! by LIF dynamics (see \ref population_algorithm).
		virtual void AdaptParameters
		(	
		) = 0;

		virtual void RecalculateSolverParameters() = 0;

		virtual Rate CalculateRate() const = 0;

		//! Some  AbstractZeroLeakEquations have derived classes which keep track of refractive probability.
		//! These derived classes can overload this method, and make this amount available. For example,
		//! when rebinning this probability must be taken into account. See, e.g. RefractiveCirculantSolver.
		virtual Probability RefractiveProbability() const { return 0.0;}

	protected:

		void SetInputParameter(const InputParameterSet& set){ _p_set = &set; }

		//! concrete instances of ZeroLeakEquations need to be able to manipulate mode
		void SetMode(CirculantMode mode, AbstractCirculantSolver& solver){solver._mode = mode;}
		//! concrete instances of ZeroLeakEquations need to be able to manipulate mode
		void SetMode(CirculantMode mode, AbstractNonCirculantSolver& solver){solver._mode = mode;}

	protected:

		valarray<double>&					ArrayState	(){ return _array_state; }
		const PopulistSpecificParameter&	ParSpec		() const { return _par_spec; }
		const SpecialBins&					Bins		() const { return _bins; }
		const InputParameterSet&			Set			() const { return *_p_set; }

	private:

		friend class AbstractRebinner;

		// Upon rebinning the refractive probability that an AbstractZeroLeakEquations subclass maintains must be
		// rescaled. This is only allowed to AbstractRebinners.
		virtual void ScaleRefractiveProbability(double){}

		valarray<double>&					_array_state;
		const PopulationParameter&			_par_pop;
		const PopulistSpecificParameter&	_par_spec;
		const SpecialBins&					_bins;
		const InputParameterSet*			_p_set;
	};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ABSTRACTZEROLEAKEQUATIONS_HPP_
