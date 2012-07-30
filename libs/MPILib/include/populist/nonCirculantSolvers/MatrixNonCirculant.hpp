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
#ifndef MPILIB_POPULIST_NONCIRCULANTSOLVERS_MATRIXNONCIRCULANT_HPP_
#define MPILIB_POPULIST_NONCIRCULANTSOLVERS_MATRIXNONCIRCULANT_HPP_

#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>
namespace MPILib {
namespace populist {
namespace nonCirculantSolvers {

//! MatrixNonCirculant
//! This is a circulant solver, which calculates the exponentiated non circulant matrix directly
//! The aim is to get rid of numerical artefacts that use the analytic expression for the non-circulant
//! solution directly.
class MatrixNonCirculant: public AbstractNonCirculantSolver {
public:

	MatrixNonCirculant();

	//! destructor
	virtual ~MatrixNonCirculant();

	/**
	 * Execute the algorithm over a given time step,
	 * for the currently valid number of bins, for the excitatory parameters.
	 *
	 * This is the most straightforward version of the algorithm: simply create a single
	 * row that contains exp Lt.
	 * Expectation is that this will be slower than the standard NonCirculantSolver,
	 * but that it is independent of the number of input populations to a first approximation,
	 * because the time to set up the matrix row is shorter than the time to carry out the matrix
	 * multiplication.
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	virtual void ExecuteExcitatory(Number n_bins, Time tau);

	/**
	 * Execute the algorithm over a given time step,
	 * for the currently valid number of bins, for the inhibitory parameters
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	virtual void ExecuteInhibitory(Number n_bins, Time tau);

	/**
	 * Clone method
	 * @return A clone of MatrixNonCirculant
	 */
	virtual MatrixNonCirculant* Clone() const;

	/**
	 * Before every solution step the input parameters may have changed, and need to be adapted
	 * @param array_state State array containing the population density
	 * @param input_set Current input parameters, see InputParameterSet for documentation,
	 * @param epsilon epsilon precision value overruling EPS_J_CIRC_MAX, when set to zero, EPS_J_CIRC_MAX is used
	 */
	virtual void Configure(std::valarray<double>& array_state,
			const parameters::InputParameterSet& input_set, double epsilon = 0);

private:

	/**
	 * Store the array state
	 */
	std::valarray<Potential> _matrix_row;
};
} /* namespace nonCirculantSolvers */
} /* namespace populist */
} /* namespace MPILib */
#endif // include guard MPILIB_POPULIST_NONCIRCULANTSOLVERS_MATRIXNONCIRCULANT_HPP_
