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
#ifndef MPILIB_ALGORITHMS_ABSTRACTREBINNER_HPP_
#define MPILIB_ALGORITHMS_ABSTRACTREBINNER_HPP_

#include  <valarray>
#include <MPILib/include/algorithm/AbstractZeroLeakEquations.hpp>
#include  <MPILib/include/BasicTypes.hpp>




namespace MPILib {
namespace algorithm {


	//! AbstractRebinner: Abstract base class for rebinning algorithms.
	//! 
	//! Rebinning algorithms serve to represent the density grid in the original grid, which is smaller
	//! than the current grid, because grids are expanding over time. Various ways of rebinning are conceivable
	//! and it may be necessary to compare different rebinning algorithms in the same program. The main simulation
	//! step in PopulationGridController only needs to know that there is a rebinning algorithm.
	class AbstractRebinner
	{
	public:

		//!
		virtual ~AbstractRebinner() = 0;

		//! Configure 
		//! Here the a reference to the bin contenets, as well as parameters necessary for the rebinning are set
		virtual bool Configure
			(	
				std::valarray<double>&,
				Index,               //!< reversal bin,
				Index,               //!< reset bin
				Number,              //!< number of  bins before rebinning
				Number               //!< number of  bins after rebinning
			) = 0;

		//! every rebinner can do a rebin after it has been configured
		//! some rebinners need to take refractive probability into account
		virtual bool Rebin(AbstractZeroLeakEquations*) = 0;

		virtual AbstractRebinner* Clone() const = 0;

		//! every rebinner has a name
		virtual std::string Name() const = 0;

	protected:

		void ScaleRefractive(double, AbstractZeroLeakEquations*);
	};


}
} // end of namespace

#endif // include guard MPILIB_ALGORITHMS_ABSTRACTREBINNER_HPP_
