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
#ifndef MPILIB_POPULIST_NONCIRCULANTSOLVERS_LIMITEDNONCIRCULANT_HPP_
#define MPILIB_POPULIST_NONCIRCULANTSOLVERS_LIMITEDNONCIRCULANT_HPP_

#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>

namespace MPILib {
namespace populist {
namespace nonCirculantSolvers {

/**
 *  @brief A class for the LimitedNonCirculant solver
 */
class LimitedNonCirculant: public AbstractNonCirculantSolver {
public:

	/**
	 * Default constructor
	 */
	LimitedNonCirculant();

	/**
	 * virtual destructor
	 */
	virtual ~LimitedNonCirculant(){};

	/**
	 * Execute the algorithm over a given time step,
	 * for the currently valid number of bins, for the excitatory parameters
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
	 * @return A clone of LimitedNonCirculant
	 */
	virtual LimitedNonCirculant* Clone() const;

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
};
} /* namespace nonCirculantSolvers */
} /* namespace populist */
} /* namespace MPILib */
#endif // include guard MPILIB_POPULIST_NONCIRCULANTSOLVERS_LIMITEDNONCIRCULANT_HPP_
