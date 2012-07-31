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
#ifndef MPILIB_POPULIST_CIRCULANTSOLVERS_POLYNOMIALCIRCULANT_HPP_
#define MPILIB_POPULIST_CIRCULANTSOLVERS_POLYNOMIALCIRCULANT_HPP_

#include <MPILib/include/populist/circulantSolvers/AbstractCirculantSolver.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <vector>

namespace MPILib {
namespace populist {
namespace circulantSolvers {

/**
 * This uses the short-time polynomial expansion of the analytic solution
 * The algorithm can only be used for small values of tau and will throw an
 * exception if it enters an unvalid regime.
 */
class PolynomialCirculant: public AbstractCirculantSolver {
public:

	/**
	 * default constructor
	 */
	PolynomialCirculant();

	/**
	 * virtual destructor
	 */
	virtual ~PolynomialCirculant() {
	}
	;

	/**
	 * Only concrete CirculantSolvers know how to compute their contribution. At this stage it is assumed that
	 * during configure the InputParameterSet is defined. The number of circulant bins, the number of
	 * non_circulant_areas. H_exc and alpha_exc must all be defined.
	 * @param n_bins Current number of bins that needs to be solved, cannot be larger than number of elements in state array
	 * @param tau Time through which evolution needs to be carried out, this may not be related to the absolute time of the simulation
	 * @param t_sim Irrelevant for this solver
	 */
	virtual void Execute(Number n_bins, Time tau, Time t_sim = 0);

	/**
	 * PolynomialCirculant computes how many circulant bins make sense
	 * @return The number of circulant bins that make sense
	 */
	virtual Number NrCirculant() const;

	/**
	 * Clone operator
	 * @return A clone of PolynomialCirculant
	 */
	virtual PolynomialCirculant* clone() const
	override;

	/**
	 * Some magical numbers are used in this CicrculantSolver
	 * @return Some magical numbers
	 */
	Index JMax() const;

private:
	/**
	 * Purpose: Integrate the density in the non-circulant areas. This produces the vector f^0.
	 */
	virtual void FillNonCirculantBins();

	/**
	 * Initialises the j_array
	 */
	void LoadJArray();

	/**
	 * The j array
	 */
	std::vector<double> _j_array;

};
} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_CIRCULANTSOLVERS_POLYNOMIALCIRCULANT_HPP_
