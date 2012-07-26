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
#ifndef MPILIB_POPULIST_CIRCULANTSOLVERS_ABSTRACTCIRCULANTSOLVER_HPP_
#define MPILIB_POPULIST_CIRCULANTSOLVERS_ABSTRACTCIRCULANTSOLVER_HPP_

#include <valarray>
#include <MPILib/include/populist/parameters/InputParameterSet.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <cassert>



namespace MPILib {
namespace populist {

//! In the Configure method of AbstractCirculantSolver an InputSetParameter reference is passed in. This contains
//! both an integer interpretation of the current potential jump in terms of number of bins (e.g. _H_exc) as well
//! as a floating point. DiffusionZeroLeakEquations will work with an integer version, whilst now for example
//! SingleInputZeroLeakEquations allow probability transport from one bin to a point between two bins, which requires
//! a floating point interpretation of the step size. This choice must be taken by a ZeroLeak developer. This developer must also
//! ensure that the CirculantMode and the NonCirculatMode are used consistently. Therefore also AbstractNonCirculantSolver uses this enum.
enum CirculantMode {FLOATING_POINT, INTEGER};

namespace circulantSolvers {

	//! Both AbstractCirculantSolver and AbstractNonCirculantSolver instances sometimes can cut calculations short
	//! by terminating the series \f$ e^{-t}\frac{t^k}{k!}$\f$, when it falls below a certain precision.
	inline Number NumberOfSolverTerms(Time t, double precision, Number n_max){
		double fact = 1.0;
		double et = exp(-t);
		for (Index i = 0; i < n_max; i++){
			fact *= t/i;
			if (et*fact < precision)
				return i;
		}
		return n_max;
	}



	//! AbstractCirculantSolver
	//! 
	//! This class stores the integrated non-circulant density areas (the f^0) and the computed
	//! circulant solution. How they are computed is a matter for the derived classes. There are
	//! also references to the current input parameters and the density profile, so that a circulant
	//! algorithm always has access to the variables. The Index method allows the PopulationgridControlller
	//! to access the calculated circulant solution.
	class AbstractCirculantSolver {

	public:

		//! default constructor, the mode can be set here, but when used in AbstractZeroLeakEquations they can and generally will overrule
		//! the choice made in the constructor. 
		AbstractCirculantSolver
		(
			CirculantMode = INTEGER,
			double precision = 0
		);

		//! virtual destructor, required for pure virtual base class
		virtual ~AbstractCirculantSolver() = 0;

		//! Configure the calculation with the current InputParameterSet
		//! For running the AbstractCirculantSolver in INTEGER mode, _n_circ_exc, _n_noncirc_exc and _H_exc must be properly
		//! defined, or the result will be undefined. See the doucmentation of LIFConvertor and InputParameterSet for information.
		virtual bool Configure
		(
			std::valarray<double>*,
			const parameters::InputParameterSet&
		);

		//! Only concrete CirculantSolvers know how to compute their contribution. At this stage it is assumed that
		//! during configure the InputParameterSet is defined. The number of circulant bins, the number of non_circulant_areas.
		//! H_exc and alpha_exc must all be defined.
		virtual void Execute
		(
			Number,			//!< Current number of bins that needs to be solved, cannot be larger than number of elements in state array
			Time,			//!< Time through which evolution needs to be carried out, this may not be related to the absolute time of the simulation
			Time = 0        //!< Where the absolute time is necessary, it can be passed here
		) = 0;

		//! A CirculantSolver knows that sometimes not all circulant bins need to be taken into account
		virtual Number NrCirculant() const;

		//! Clone function
		virtual AbstractCirculantSolver* Clone() const = 0;

		//! Access to the circulant contribution. Tryimg to access beyond NrCirculant() - 1 is undefined
		double operator[](Index) const;

		//! The total amount of density which has ended up in the circulant bins
		Density IntegratedFlux() const;

		//! zero leak equations must sometimes interfere with solvers, for example by setting mode
		friend class AbstractZeroLeakEquations;

		//! It wil transfer the contributions calculated in the AbstractCirculantSolver, which knows nothing about the neuronal
		//! parameters to the state array. Therefor it needs to receive the index of the current reset bin.
		virtual void AddCirculantToState(Index);

		//! This is a hint for AbstractZeroLeakEquations developers, whether in their implementation the Circulantsolver
		//! should be called before of after the NonCirculantSolver. As an example, CirculantSolver should be called
		//! before NonCirculantSolver, but RefractiveCirculantSolver should be called after NonCirculantSolver. In LIFZeroEquations
		//! this hint allows generic code which delegates the decision which to call first to the CirculantSolver in question.
		//! The RefractiveCirculant strategy is made default.
		virtual bool BeforeNonCirculant() {return true;}

		//! Some CirculantSolvers hold refractive probability, i.e. probability that is not represented in the current state.
		//! If so, this method needs to be overloaded
		virtual Probability RefractiveProbability() const {return 0.0;}

		//! Some rebinners need to rescale the probability held in the queue after rebinning.
		virtual void ScaleProbabilityQueue(double){}

		void setMode(CirculantMode mode){
			_mode=mode;
		}
	protected:

		void FillNonCirculantBins();

		Number	_n_bins;				
		Time	_tau;

		//! The fill algorithms store the probability density integrated per non circulant areas in array_rho. The index runs
		//! from the threshold backwards, i.e. the non circulant area bordering the threshold is area 0, conform Equation 20 in (de Kamps, 2006)

		std::valarray<double>			_array_rho;			// integrated density in the non-circulant areas (f^0)
		std::valarray<double>			_array_circulant;	// storage array for the calculated circulant solution
		std::valarray<double>*			_p_array_state = nullptr;		// pointer to the probability density array

		const parameters::InputParameterSet*	_p_set = nullptr;      // instantaneous value of the input parameters
		double						_initial_integral = 0.0;

	private:

		void FillLinear	();
		void FillFP		();

		void AddCirculantInteger(Index);
		void AddCirculantFP(Index);

		CirculantMode	_mode;

	};

	inline AbstractCirculantSolver::~AbstractCirculantSolver()
	{
	}

	inline double AbstractCirculantSolver::operator [](Index index) const
	{
		return _array_circulant[index];
	}

	inline void AbstractCirculantSolver::AddCirculantToState(Index i_reversal)
	{
		if (_mode == INTEGER)
			this->AddCirculantInteger(i_reversal);
		else
		{
			assert( _p_set->_n_circ_exc == static_cast<Number>((_n_bins - i_reversal)/(_p_set->_H_exc + _p_set->_alpha_exc))+1);
			this->AddCirculantFP(i_reversal);
		}
	}

	inline void AbstractCirculantSolver::AddCirculantInteger(Index i_reversal)
	{
		std::valarray<double>& array_state = *_p_array_state;
		Index i = i_reversal;
		int n_circ = static_cast<int>(_p_set->_n_circ_exc);
		for (int j = 0; j < n_circ; j++ ){
			array_state[i] += _array_circulant[j];
			i += _p_set->_H_exc;
		}
	}

	inline void AbstractCirculantSolver::AddCirculantFP(Index i_reversal)
	{
		std::valarray<double>& array_state = *_p_array_state;
		int n_circ = static_cast<int>(_p_set->_n_circ_exc);
		for (int j = 0; j < n_circ-1; j++){
			double h = (_p_set->_H_exc + _p_set->_alpha_exc)*j;
			int H = static_cast<int>(floor(h)) + i_reversal;
			double frac = h - floor(h);
			array_state[H]   += (1-frac)*_array_circulant[j];
			array_state[H+1] += frac*_array_circulant[j];
		}
		array_state[_n_bins-1]   += _array_circulant[n_circ-1];
	}
} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_CIRCULANTSOLVERS_ABSTRACTCIRCULANTSOLVER_HPP_
