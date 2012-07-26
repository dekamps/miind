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
#include <MPILib/include/utilities/Exception.hpp>
#include <MPILib/include/populist/ZeroLeakBuilder.hpp>
#include <MPILib/include/populist/circulantSolvers/CirculantSolver.hpp>
#include <MPILib/include/populist/circulantSolvers/PolynomialCirculant.hpp>
#include <MPILib/include/populist/circulantSolvers/RefractiveCirculantSolver.hpp>

#include <MPILib/include/populist/nonCirculantSolvers/NonCirculantSolver.hpp>
#include <MPILib/include/populist/nonCirculantSolvers/LimitedNonCirculant.hpp>
#include <MPILib/include/populist/nonCirculantSolvers/MatrixNonCirculant.hpp>

#include <MPILib/include/populist/zeroLeakEquations/LIFZeroLeakEquations.hpp>
#include <MPILib/include/populist/zeroLeakEquations/NumericalZeroLeakEquations.hpp>
#include <MPILib/include/populist/zeroLeakEquations/OldLifZeroLeakEquations.hpp>
#include <MPILib/include/populist/zeroLeakEquations/OneDMZeroLeakEquations.hpp>

#include <MPILib/include/populist/zeroLeakEquations/SingleInputZeroLeakEquations.hpp>

namespace MPILib {
namespace populist {

ZeroLeakBuilder::ZeroLeakBuilder
(
	Number&						n_bins,		
	valarray<Potential>&		array_state,
	Potential&					checksum,	
	SpecialBins&				bins,		
	parameters::PopulationParameter&		par_pop,
	parameters::PopulistSpecificParameter&	par_spec,
	Potential&					delta_v
):
_n_bins(n_bins),
_array_state(array_state),
_checksum(checksum),
_bins(bins),
_par_pop(par_pop),
_par_spec(par_spec),
_delta_v(delta_v)
{
}


std::shared_ptr<AbstractZeroLeakEquations> ZeroLeakBuilder::GenerateZeroLeakEquations
( 
	const string&									zeroleakequations_name,
	const string&									circulant_solver_name, 
	const string&									noncirculant_solver_name
)
{
	std::shared_ptr<circulantSolvers::AbstractCirculantSolver> p_circ;
	if ( circulant_solver_name == "CirculantSolver" )
		p_circ = std::shared_ptr<circulantSolvers::AbstractCirculantSolver>(new circulantSolvers::CirculantSolver);
	else
		if (circulant_solver_name  == "PolynomialCirculant" )
			p_circ = std::shared_ptr<circulantSolvers::AbstractCirculantSolver>(new circulantSolvers::PolynomialCirculant );
		else 
			if (circulant_solver_name == "RefractiveCirculantSolver")
				p_circ = std::shared_ptr<circulantSolvers::AbstractCirculantSolver>(new circulantSolvers::RefractiveCirculantSolver(_par_pop._tau_refractive));
			else
				throw utilities::Exception("Unknown Circulant");
		

	std::shared_ptr< nonCirculantSolvers::AbstractNonCirculantSolver> p_noncirc;
	if ( noncirculant_solver_name == "NonCirculantSolver" )
		p_noncirc = std::shared_ptr< nonCirculantSolvers::AbstractNonCirculantSolver>(new  nonCirculantSolvers::NonCirculantSolver );
	else
		if (noncirculant_solver_name == "LimitedNonCirculant" )
			p_noncirc = std::shared_ptr< nonCirculantSolvers::AbstractNonCirculantSolver>(new  nonCirculantSolvers::LimitedNonCirculant);
		else
			if (noncirculant_solver_name == "MatrixNonCirculant" )
				p_noncirc = std::shared_ptr< nonCirculantSolvers::AbstractNonCirculantSolver>(new  nonCirculantSolvers::MatrixNonCirculant);
			else
				throw utilities::Exception("Unknown NonCirculant solver");

	std::shared_ptr<AbstractZeroLeakEquations> p_ret;
	if (zeroleakequations_name == "NumericalZeroLeakEquations"){
		p_ret = std::shared_ptr<NumericalZeroLeakEquations>
				(
					new NumericalZeroLeakEquations
					(
						_n_bins,
						_array_state,
						_checksum,
						_bins,
						_par_pop,	
						_par_spec,	
						_delta_v
					)
				);
		return p_ret;
	}

	if (zeroleakequations_name == "LIFZeroLeakEquations"){
		p_ret = std::shared_ptr<LIFZeroLeakEquations>
				(
					new LIFZeroLeakEquations
					(
						_n_bins,
						_array_state,
						_checksum,
						_bins,
						_par_pop,	
						_par_spec,	
						_delta_v,
						*p_circ,
						*p_noncirc
					)
				);
		return p_ret;
	}

	if (zeroleakequations_name == "OldLIFZeroLeakEquations"){

		// This choice will overule the choice for the NonCirculantSolver
		p_noncirc = std::shared_ptr< nonCirculantSolvers::AbstractNonCirculantSolver>(new  nonCirculantSolvers::NonCirculantSolver(INTEGER) );

		p_ret = std::shared_ptr<LIFZeroLeakEquations>
				(
					new OldLIFZeroLeakEquations
					(
						_n_bins,
						_array_state,
						_checksum,
						_bins,
						_par_pop,	
						_par_spec,	
						_delta_v,
						*p_circ,
						*p_noncirc
					)
				);
		return p_ret;
	}
	if (zeroleakequations_name == "OneDMZeroLeakEquations" ){
		p_ret = std::shared_ptr<OneDMZeroLeakEquations>
				(
					new OneDMZeroLeakEquations
					(
						_n_bins,
						_array_state,
						_checksum,
						_bins,
						_par_pop,	
						_par_spec,	
						_delta_v
						)
				);
		return p_ret;
	}

	if (zeroleakequations_name == "SingleInputZeroLeakEquations" ){
		p_ret = std::shared_ptr<SingleInputZeroLeakEquations>
				(
					new SingleInputZeroLeakEquations
					(
						_n_bins,
						_array_state,
						_checksum,
						_bins,
						_par_pop,	
						_par_spec,	
						_delta_v,
						*p_circ,
						*p_noncirc
					)
				);

		return p_ret;
	}

	throw utilities::Exception("Unknown ZeroLeak type specified");
}

} /* namespace populist */
} /* namespace MPILib */
	
