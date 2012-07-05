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
#ifndef MPILIB_POPULIST_POPULISTSPECIFICPARAMETER_HPP_
#define MPILIB_POPULIST_POPULISTSPECIFICPARAMETER_HPP_

//#include <MPILib/include/populist/AbstractRebinner.hpp>
#include <MPILib/include/BasicTypes.hpp>


#include <MPILib/include/populist/AbstractRateComputation.hpp>
#include <MPILib/include/populist/InitialDensityParameter.hpp>
#include <boost/shared_ptr.hpp>

namespace MPILib {
namespace populist {

class AbstractRebinner;

	//! These are parameters necessary for the configuration of a PopulistAlgorithm and OneDMAlgorithm

	//! and not neuronal parameters. Number of bins, but also the algorithms for calculating
	//! the circulant and the non-circulant solutions are specified here.
	//! One important parameter that is calculated here is the maximum number of
	//! grid points that emerge from the initial number of bins specified by the user,
	//! v_min and the expansion factor
	class PopulistSpecificParameter{

	public:

		//! default constructor
		PopulistSpecificParameter();

		//! copy constructor
		PopulistSpecificParameter
		(
			const PopulistSpecificParameter&
		);

		//! constructor
		PopulistSpecificParameter
		(
			Potential,								//!< minimum potential of the grid, (typically negative or below the reversal potential
			Number,									//!< initial number of bins
			Number,									//!< number of bins that is added after one zero-leak evaluation
			const InitialDensityParameter&,			//!< gaussian (or delta-peak) initial density profile
			double,									//!< expansion factor
			const std::string&	zeroleakequation_name	= "NumericalZeroLeakEquations",		//!< The algorithm for solving the zero leak equations (see documentation at \ref AbstractZeroLeakequations if you want to modify the default choice)
			const std::string&	circulant_solver_name	= "CirculantSolver",				//!< The algorithm for solving the circulant equations (see documentation at \ref AbstractCirculant if you want to use a modofied version of this algorithm)
			const std::string&	noncirculant_solver_name= "NonCirculantSolver",				//!< The algorithm for solving the circulant equations (see documentation at \ref AbstractCirculant if you want to use a modofied version of this algorithm)
			const AbstractRebinner*	          = 0,  //!< Use when investigating alternatives to the standard rebinner, which InterpolationRebinner
			const AbstractRateComputation*    = 0	//!< Use when investigating alternatives to the standard rate computation, which is IntegralRateComputation    
		);

		//! destructor 
		virtual ~PopulistSpecificParameter();

		//! copy operator
		PopulistSpecificParameter&
			operator=
			(
				const PopulistSpecificParameter&
			);

		virtual PopulistSpecificParameter* Clone() const;

		//! Minumum Potential of Grid at initialization time
		Potential VMin() const;
		
		//! Number of bins at initialization time
		Number    NrGridInitial() const;

		//! Number of bins to be added, during evolution (almost always 1)
		Number    NrAdd() const;

		//! Maximum number of grid points that can result from the initial number of points and the
		//! expansion factor
		Number	  MaxNumGridPoints() const;

		//! Initial probability density profile
		InitialDensityParameter InitialDensity() const;

		//! Expansion factor
		double ExpansionFactor() const;

		const AbstractRebinner& Rebin() const;

		const AbstractRateComputation& RateComputation() const;

		std::string ZeroLeakName() const {return _name_zeroleak; }

		std::string CirculantName() const { return _name_circulant; }

		std::string NonCirculantName() const {return _name_noncirculant; }


	private:

		Potential							_v_min;
		Number								_n_grid_initial;
		Number								_n_add;
		InitialDensityParameter			_par_dens;
		double								_fact_expansion;
		std::string								_name_zeroleak;
		std::string								_name_circulant;
		std::string								_name_noncirculant;

		boost::shared_ptr<AbstractRebinner>			_p_rebinner;
		boost::shared_ptr<AbstractRateComputation>	_p_rate;
	};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_POPULISTSPECIFICPARAMETER_HPP_
