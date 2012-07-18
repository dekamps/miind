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
#ifndef MPILIB_POPULIST_PARAMETERS_POPULISTSPECIFICPARAMETER_HPP_
#define MPILIB_POPULIST_PARAMETERS_POPULISTSPECIFICPARAMETER_HPP_

#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/InitialDensityParameter.hpp>
#include <string>
#include <memory>

namespace MPILib {
namespace populist {

class AbstractRebinner;
class AbstractRateComputation;

//! These are parameters necessary for the configuration of a PopulistAlgorithm and OneDMAlgorithm

//! and not neuronal parameters. Number of bins, but also the algorithms for calculating
//! the circulant and the non-circulant solutions are specified here.
//! One important parameter that is calculated here is the maximum number of
//! grid points that emerge from the initial number of bins specified by the user,
//! v_min and the expansion factor
class PopulistSpecificParameter {

public:

	/**
	 * default constructor
	 */
	PopulistSpecificParameter();

	/**
	 * copy constructor
	 * @param Another PopulistSpecificParameter
	 */
	PopulistSpecificParameter(const PopulistSpecificParameter&);

	/**
	 * constructor
	 * @param v_min minimum potential of the grid, (typically negative or below the reversal potential)
	 * @param n_grid_initial initial number of bins
	 * @param n_add number of bins that is added after one zero-leak evaluation
	 * @param par_dens gaussian (or delta-peak) initial density profile
	 * @param fact_expansion expansion factor
	 * @param name_zeroleak The algorithm for solving the zero leak equations (see documentation at \ref AbstractZeroLeakequations if you want to modify the default choice)
	 * @param name_circulant The algorithm for solving the circulant equations (see documentation at \ref AbstractCirculant if you want to use a modified version of this algorithm)
	 * @param name_noncirculant The algorithm for solving the non circulant equations (see documentation at \ref AbstractNonCirculant if you want to use a modified version of this algorithm)
	 * @param p_rebinner Use when investigating alternatives to the standard rebinner, which InterpolationRebinner
	 * @param p_rate Use when investigating alternatives to the standard rate computation, which is IntegralRateComputation
	 */
	PopulistSpecificParameter(Potential v_min, Number n_grid_initial,
			Number n_add, const InitialDensityParameter& par_dens,
			double fact_expansion, const std::string& name_zeroleak= "NumericalZeroLeakEquations",
			const std::string& name_circulant  = "CirculantSolver" ,
			const std::string& name_noncirculant = "NonCirculantSolver",
			const AbstractRebinner* p_rebinner = nullptr,
			const AbstractRateComputation* p_rate = nullptr);

	/**
	 * destructor
	 */
	virtual ~PopulistSpecificParameter();

	/**
	 * copy operator
	 * @param another PopulistSpecificParameter
	 * @return a copy of this
	 */
	PopulistSpecificParameter&
	operator=(const PopulistSpecificParameter&);

	/**
	 * clones this class
	 * @return A clone of this class
	 */
	virtual PopulistSpecificParameter* Clone() const;

	/**
	 * Getter for minimum potential
	 * @return Minimum Potential of Grid at initialization time
	 */
	Potential getVMin() const;

	/**
	 * Getter for number of bins
	 * @return Number of bins at initialization time
	 */
	Number getNrGridInitial() const;

	/**
	 * Getter for number of bins to be added
	 * @return Number of bins to be added, during evolution (almost always 1)
	 */
	Number getNrAdd() const;

	/**
	 * Getter for maximum number of grid points
	 * @return Maximum number of grid points that can result from the initial number of points and the expansion factor
	 */
	Number getMaxNumGridPoints() const;

	/**
	 * Getter for initial probability density profile
	 * @return Initial probability density profile
	 */
	InitialDensityParameter getInitialDensity() const;

	/**
	 * Getter for the expansion factor
	 * @return Expansion factor
	 */
	double getExpansionFactor() const;

	/**
	 * Getter for the AbstractRebinner
	 * @return The AbstractRebinner
	 */
	const AbstractRebinner& getRebin() const;

	/**
	 * Getter for the AbstractRateComputation
	 * @return The AbstractRateComputation
	 */
	const AbstractRateComputation& getRateComputation() const;

	/**
	 * Getter for the name of the algorithm for solving the zero leak equations
	 * @return The name of the algorithm for solving the zero leak equations
	 */
	std::string getZeroLeakName() const;
	/**
	 * Getter for the name of the algorithm for solving the circulant equations
	 * @return The name of the algorithm for solving the circulant equations
	 */
	std::string getCirculantName() const;
	/**
	 * Getter for the name of the algorithm for solving the non circulant equations
	 * @return The name of the algorithm for solving the non circulant equations
	 */
	std::string getNonCirculantName() const;

private:

	/**
	 * minimum potential of the grid
	 */
	Potential _v_min = 0.0;
	/**
	 * initial number of bins
	 */
	Number _n_grid_initial = 0;
	/**
	 * number of bins that is added after one zero-leak evaluation
	 */
	Number _n_add = 0;
	/**
	 * gaussian (or delta-peak) initial density profile
	 */
	InitialDensityParameter _par_dens = InitialDensityParameter(0.0, 0.0);
	/**
	 * expansion factor
	 */
	double _fact_expansion = 0.0;
	/**
	 * The name of the algorithm for solving the zero leak equations
	 */
	std::string _name_zeroleak;
	/**
	 * The name of the algorithm for solving the circulant equations
	 */
	std::string _name_circulant;
	/**
	 * The name of the algorithm for solving the non circulant equations
	 */
	std::string _name_noncirculant;

	/**
	 * Use when investigating alternatives to the standard rebinner
	 */
	std::shared_ptr<AbstractRebinner> _p_rebinner;
	/**
	 * Use when investigating alternatives to the standard rate computation
	 */
	std::shared_ptr<AbstractRateComputation> _p_rate;
};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_PARAMETERS_POPULISTSPECIFICPARAMETER_HPP_
