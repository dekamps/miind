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
#ifndef MPILIB_POPULIST_NONCIRCULANTSOLVERS_ABSTRACTNONCIRCULANTSOLVER_HPP_
#define MPILIB_POPULIST_NONCIRCULANTSOLVERS_ABSTRACTNONCIRCULANTSOLVER_HPP_

#include <valarray>
#include <cassert>
#include <MPILib/include/populist/CirculantMode.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/InputParameterSet.hpp>

namespace MPILib {
namespace populist {
namespace nonCirculantSolvers {

/**
 * @brief A base class for all non-circulant solvers.
 *
 * The idea was that they can be exchanged during run time to investigate changes in the algorithm
 * on the network level. This seems less important now. The base class stores an array of the
 * form \f$ (e^{-\tau}, e^{-\tau}tau, e^{-\tau}\frac{\tau}{2}, \cdots e^{-\tau}\frac{\tau^k}{k!}, \cdots ) \f$ .
 * The sequence is broken off when the last term is smaller than  EPS_J_CIRC_MAX.
 * The number of bins is efficacy in terms of number of bins.
 */
class AbstractNonCirculantSolver {
public:

	/**
	 * The mode can be set on upon construction, but when used in AbstractZeroLeakEquations
	 * instances the mode can and generally will be overruled
	 * @param mode The CirculantMode the solver is set to
	 */
	AbstractNonCirculantSolver(CirculantMode mode);

	/**
	 * Before every solution step the input parameters may have changed, and need to be adapted
	 * @param array_state State array containing the population density
	 * @param input_set Current input parameters, see InputParameterSet for documentation,
	 * @param epsilon epsilon precision value overruling EPS_J_CIRC_MAX, when set to zero, EPS_J_CIRC_MAX is used
	 */
	virtual void Configure(std::valarray<double>& array_state,
			const parameters::InputParameterSet& input_set, double epsilon = 0);

	/**
	 * Execute the algorithm over a given time step,
	 * for the currently valid number of bins, for the excitatory parameters
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	virtual void ExecuteExcitatory(Number n_bins, Time tau)=0;

	/**
	 * Execute the algorithm over a given time step,
	 * for the currently valid number of bins, for the inhibitory parameters
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	virtual void ExecuteInhibitory(Number n_bins, Time tau)= 0;

	/**
	 * virtual destructor for correct removal of allocated resources
	 */
	virtual ~AbstractNonCirculantSolver() {
	}
	;

	/**
	 * Clone method
	 * @return A clone of AbstractNonCirculantSolver
	 */
	virtual AbstractNonCirculantSolver* Clone() const = 0;

	/**
	 * Return the maximum number of terms available in array _array_factor, after the most recent
	 * run of InitializeArrayFactor. Only valid after InitializeArrayFactor was run.
	 * @return The maximum number of terms available in array _array_factor
	 */
	Number NumberFactor() const {
		assert(_j_circ_max > 0);
		return _j_circ_max;
	}

	/**
	 * Setter for the mode
	 * @param mode The new circulant mode
	 */
	void setMode(CirculantMode mode) {
		_mode = mode;
	}

protected:
	/**
	 *  Initializes an array with values \f$ e^{-tau}, \tau e^{-\tau}, \cdots \frac{\tau^k}{k!}e^{-\tau} \f$,
	 *  but breaks the series if the terms become smaller than epsilon, which is set at Configure.
	 *  There are then less terms than requested, the number of terms actually calculated is returned.
	 *  The results are stored in _array_factor. Attempts to access _array_factor beyond position NumberFactor() - 1
	 *  see below result in undefined behaviour. This number of elements in the factor array is usually
	 *  calculated in one of the Execute* functions so would be known there. In general clients are advised
	 *   not to use this function.
	 * @param tau Time by which to evolve
	 * @param n_non_circulant maximum number of terms of the factor array
	 */
	virtual void InitializeArrayFactor(Time tau, Number n_non_circulant);

	/**
	 * The array with the factors
	 */
	std::valarray<double> _array_factor;
	/**
	 * Pointer to the state array containing the population density
	 */
	std::valarray<double>* _p_array_state = nullptr;
	/**
	 * Pointer to the input set
	 */
	const parameters::InputParameterSet* _p_input_set;
	/**
	 * Precision value overruling EPS_J_CIRC_MAX
	 */
	double _epsilon;
	/**
	 *  maximum j for which \f$ exp(-tau) tau^k/k! \f$ is relevant. Helps to cut computation short.
	 */
	int _j_circ_max = -1;
	/**
	 * The circulant mode of the solver
	 */
	CirculantMode _mode;

};

} /* namespace nonCirculantSolvers */
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_NONCIRCULANTSOLVERS_ABSTRACTNONCIRCULANTSOLVER_HPP_
