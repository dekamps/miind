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
#ifndef MPILIB_POPULIST_NONCIRCULANTSOLVERS_NONCIRCULANTSOLVER_HPP_
#define MPILIB_POPULIST_NONCIRCULANTSOLVERS_NONCIRCULANTSOLVER_HPP_

#include <valarray>
#include <MPILib/include/populist/circulantSolvers/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/InputParameterSet.hpp>

namespace MPILib {
namespace populist {
namespace nonCirculantSolvers {

/**
 * The algorithm that runs the Poisson master process for probability density corresponding to neurons
 * that have not yet passed threshold. It should never be run in INTEGER mode, a mode which has been
 * left in for the reproduction of old paper results. The algorithm can be created, but nothing will
 * happen at that point. Once it has been configured, telling it which valarray to run on, and what the
 * current input parameters are, it can be run by calling apply on a specified number of bins.
 *
 */
class NonCirculantSolver: public AbstractNonCirculantSolver {
public:

	/**
	 * 	Only mode selection in the constructor, hard work is done in Configure function of the
	 * 	base class. If INTEGER mode is chosen, the _H_exc and _H_inh are used as a parameter,
	 * 	for FLOATING_POINT _h_exc and _h_inh are used.
	 * @param mode The CirculantMode of this Solver
	 */
	NonCirculantSolver(CirculantMode mode = FLOATING_POINT);

	/**
	 * Carry out one excitatory step for the number of bins given, using the input parameters that
	 * were set in the Configure method of the base class.
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	virtual void ExecuteExcitatory(Number n_bins, Time tau);

	/**
	 * Carry out one inhibitory step for the number of bins given, using the input parameters that
	 * were set in the Configure method of the base class
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	virtual void ExecuteInhibitory(Number n_bins, Time tau);

	/**
	 * Simply add the total amount of probability present in the density array for the current number of bins
	 * @param number_of_bins current number of bins
	 * @return The total amount of probability present in the density array
	 */
	double Integrate(Number number_of_bins) const;

	/**
	 * Clone method
	 * @return A clone of NonCirculantSolver
	 */
	virtual NonCirculantSolver* Clone() const;

private:

	/**
	 * Helper Function for ExecuteExcitatory for INTEGER mode
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	void ExecuteIntegerExcitatory(Number n_bins, Time tau);
	/**
	 * Helper Function for ExecuteInhibitory for INTEGER mode
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	void ExecuteIntegerInhibitory(Number n_bins, Time tau);
	/**
	 * Helper Function for ExecuteInhibitory for FLOATING_POINT mode
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	void ExecuteFractionExcitatory(Number n_bins, Time tau);
	/**
	 * Helper Function for ExecuteInhibitory for FLOATING_POINT mode
	 * @param n_bins Number of bins for which the solver must operate, i.e. the current number of bins. May not exceed number of elements in state array.
	 * @param tau Time by which to evolve
	 */
	void ExecuteFractionInhibitory(Number n_bins, Time tau);

};
// end of NonCirculantSolver

} /* namespace nonCirculantSolvers */
} /* namespace populist */
} /* namespace MPILib */
#endif // include guard MPILIB_POPULIST_NONCIRCULANTSOLVERS_NONCIRCULANTSOLVER_HPP_
