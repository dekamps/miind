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
#ifndef MPILIB_POPULIST_CIRCULANTSOLVER_HPP_
#define MPILIB_POPULIST_CIRCULANTSOLVER_HPP_

#include <valarray>
#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/populist/AbstractCirculantSolver.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/InputParameterSet.hpp>
#include <MPILib/include/populist/VArray.hpp>

using NumtoolsLib::C_Matrix;
using NumtoolsLib::D_Matrix;

namespace MPILib {
namespace populist {

	//! CirculantSolver
	//! This is the literal implementation of the analytic solution from (de Kamps, 2006)
	//! It is not efficient and should probably not be used in realistic applications, but
	//! is important in the benchmarking of other circulant solvers
	class CirculantSolver : public AbstractCirculantSolver
	{
	public:

		CirculantSolver(CirculantMode = INTEGER);

		virtual void Execute
		(
			Number,
			Time,
			Time = 0 //!< Irrelevant for this solver
		);


		double Integrate(Number) const;

		double Flux
		(
			Number, 
			Time
		) const;

		//! Clone operation
		CirculantSolver* Clone() const;

		//! 
		virtual bool BeforeNonCirculant() {return true;}

	private:

		void CalculateInnerProduct();


		Index				_index_reversal_bin;
		VArray				_array_V;
	};
} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_CIRCULANTSOLVER_HPP_
