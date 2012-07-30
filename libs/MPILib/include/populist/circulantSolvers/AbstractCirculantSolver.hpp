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
#include <MPILib/include/populist/CirculantMode.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <cassert>

namespace MPILib {
namespace populist {
namespace circulantSolvers {

/**
 * Both AbstractCirculantSolver and AbstractNonCirculantSolver instances sometimes can cut calculations short
 * by terminating the series \f$ e^{-t}\frac{t^k}{k!}$\f$, when it falls below a certain precision.
 * @param t The parameter t
 * @param precision The precision
 * @param n_max The max number of terms
 * @return The max number of terms considered
 */
inline Number NumberOfSolverTerms(Time t, double precision, Number n_max) {
	double fact = 1.0;
	double et = exp(-t);
	for (Index i = 0; i < n_max; i++) {
		fact *= t / i;
		if (et * fact < precision)
			return i;
	}
	return n_max;
}

/**
 * @brief This class stores the integrated non-circulant density areas (the f^0) and the computed
 * circulant solution.
 *
 *  This class stores the integrated non-circulant density areas (the f^0) and the computed
 *  also references to the current input parameters and the density profile, so that a circulant
 *  algorithm always has access to the variables. The Index method allows the PopulationgridControlller
 *  to access the calculated circulant solution.
 */
class AbstractCirculantSolver {

public:

	/**
	 * default constructor, the mode can be set here, but when used in AbstractZeroLeakEquations they
	 * can and generally will overrule the choice made in the constructor.
	 * @param mode the circulant mode
	 * @param precision The precicion
	 */
	AbstractCirculantSolver(CirculantMode mode = INTEGER, double precision = 0);

	/**
	 * virtual destructor
	 */
	virtual ~AbstractCirculantSolver(){};

	/**
	 * Configure the calculation with the current InputParameterSet
	 * For running the AbstractCirculantSolver in INTEGER mode, _n_circ_exc, _n_noncirc_exc
	 * and _H_exc must be properly defined, or the result will be undefined.
	 * See the doucmentation of LIFConvertor and InputParameterSet for information.
	 * @param p_array_state A pointer to the state of the array
	 * @param set The input parameter set
	 */
	virtual void Configure(std::valarray<double>* p_array_state,
			const parameters::InputParameterSet&);

	/**
	 * Only concrete CirculantSolvers know how to compute their contribution. At this stage it is assumed that
	 * during configure the InputParameterSet is defined. The number of circulant bins, the number of
	 * non_circulant_areas. H_exc and alpha_exc must all be defined.
	 * @param n_bins Current number of bins that needs to be solved, cannot be larger than number of elements in state array
	 * @param tau Time through which evolution needs to be carried out, this may not be related to the absolute time of the simulation
	 * @param t_sim Where the absolute time is necessary, it can be passed here
	 */
	virtual void Execute(Number n_bins, Time tau, Time t_sim = 0) = 0;

	/**
	 * A CirculantSolver knows that sometimes not all circulant bins need to be taken into account
	 * @return the number of circulant bins
	 */
	virtual Number NrCirculant() const;

	/**
	 * Clone operation
	 * @return A clone of the AbstractCirculantSolver
	 */
	virtual AbstractCirculantSolver* clone() const = 0;

	/**
	 * Access to the circulant contribution. Tryimg to access beyond NrCirculant() - 1 is undefined
	 * @param index The index of the accessed element
	 * @return The value of the accesssed element
	 */
	double operator[](Index index) const;

	/**
	 * Compute the total amount of density which has ended up in circulant bins
	 * This must much the amount of density that has left the non-circulant bins
	 * @return The total amount of density
	 */
	Density IntegratedFlux() const;

	/**
	 * It will transfer the contributions calculated in the AbstractCirculantSolver, which knows nothing about
	 * the neuronal parameters to the state array. Therefore it needs to receive the index of the current reset bin.
	 * @param i_reversal the reversal index
	 */
	virtual void AddCirculantToState(Index i_reversal);

	/**
	 * This is a hint for AbstractZeroLeakEquations developers, whether in their implementation the
	 * Circulantsolver should be called before of after the NonCirculantSolver. As an example,
	 * CirculantSolver should be called before NonCirculantSolver, but RefractiveCirculantSolver
	 * should be called after NonCirculantSolver. In LIFZeroEquations this hint allows generic code
	 * which delegates the decision which to call first to the CirculantSolver in question.
	 * The RefractiveCirculant strategy is made default.
	 * @return false if the NonCirculant must be executed first
	 */
	virtual bool BeforeNonCirculant() {
		return true;
	}

	/**
	 * Some CirculantSolvers hold refractive probability, i.e. probability that is not represented in the current state.
	 * If so, this method needs to be overloaded
	 * @return In the default case 0
	 */
	virtual Probability RefractiveProbability() const {
		return 0.0;
	}

	/**
	 *  Some rebinners need to rescale the probability held in the queue after rebinning.
	 * @param scale The scale factor
	 */
	virtual void ScaleProbabilityQueue(double scale) {
	}

	/**
	 * Setter for the CirculantMode
	 * @param mode The new Circulant Mode the solver is set to.
	 */
	void setMode(CirculantMode mode) {
		_mode = mode;
	}
protected:

	/**
	 * The fill algorithms store the probability density integrated per non circulant areas in area_rho.
	 * The index runs from the threshold backwards, i.e. the non circulant area bordering the threshold is area 0,
	 * conform Equation 20 in (de Kamps, 2006)
	 */
	void FillNonCirculantBins();

	/**
	 * The number of bins
	 */
	Number _n_bins;
	/**
	 * The parameter tau
	 */
	Time _tau;

	/**
	 * The fill algorithms store the probability density integrated per non circulant areas in array_rho.
	 * The index runs from the threshold backwards, i.e. the non circulant area bordering the threshold
	 * is area 0, conform Equation 20 in (de Kamps, 2006)
	 * integrated density in the non-circulant areas (f^0)
	 */
	std::valarray<double> _array_rho;
	/**
	 *  storage array for the calculated circulant solution
	 */
	std::valarray<double> _array_circulant;
	/**
	 * pointer to the probability density array
	 */
	std::valarray<double>* _p_array_state = nullptr;
	/**
	 * instantaneous value of the input parameters
	 */
	const parameters::InputParameterSet* _p_set = nullptr;
	/**
	 * The value of the initial Integral
	 */
	double _initial_integral = 0.0;

private:
	/**
	 * Purpose: Integrate the density in the non-circulant areas. This produces the vector f^0.
	 * Assumptions: _p_set->_n_noncirc_exc and _p_set->H_exc are defined and consistent
	 */
	void FillLinear ();
	/**
	 * Alternative to FillLinear if the mode is double
	 */
	void FillFP ();

	/**
	 * It will transfer the contributions calculated in the AbstractCirculantSolver, which knows nothing about
	 * the neuronal parameters to the state array. Therefore it needs to receive the index of the current reset bin.
	 * This is the method for integer mode
	 * @param i_reversal the reversal index
	 */
	void AddCirculantInteger(Index i_reversal);
	/**
	 * It will transfer the contributions calculated in the AbstractCirculantSolver, which knows nothing about
	 * the neuronal parameters to the state array. Therefore it needs to receive the index of the current reset bin.
	 * This is the method for double mode
	 * @param i_reversal the reversal index
	 */
	void AddCirculantFP(Index i_reversal);

	/**
	 * The circulant mode
	 */
	CirculantMode _mode;

};

} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_CIRCULANTSOLVERS_ABSTRACTCIRCULANTSOLVER_HPP_
