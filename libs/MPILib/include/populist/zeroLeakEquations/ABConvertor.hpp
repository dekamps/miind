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
#ifndef MPILIB_POPULIST_ABCONVERTOR_HPP_
#define MPILIB_POPULIST_ABCONVERTOR_HPP_

#ifdef WIN32
#pragma warning(disable: 4996)
#endif 

#include <iostream>
#include <valarray>

#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

#include <MPILib/include/populist/zeroLeakEquations/ABQStruct.hpp>
#include <MPILib/include/populist/zeroLeakEquations/ABScalarProduct.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>
#include <MPILib/include/populist/parameters/OneDMParameter.hpp>
#include <MPILib/include/populist/parameters/OneDMInputSetParameter.hpp>
#include <MPILib/include/populist/zeroLeakEquations/SpecialBins.hpp>
#include <MPILib/include/NodeType.hpp>

namespace MPILib {
namespace populist {

class ABConvertor {
public:

	typedef parameters::OneDMInputSetParameter SolverParameterType;
	typedef ABQStruct ScalarProductParameterType;

	ABConvertor( VALUE_REF
	SpecialBins&, parameters::PopulationParameter&,	//!< serves now mainly to communicate t_s
	parameters::PopulistSpecificParameter&, Potential&,	//!< current potential interval covered by one bin, delta_v
			Number&);

	void Configure(std::valarray<Potential>&, std::valarray<Potential>&,
			const parameters::OneDMParameter& par_onedm) {
		_param_onedm = par_onedm;
	}

	void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector);

	void AdaptParameters();

	void RecalculateSolverParameters();

	void Rebin() {
	}

	const parameters::PopulistSpecificParameter&
	PopSpecific() const;

	const parameters::OneDMInputSetParameter&
	InputSet() const;

	const parameters::PopulationParameter& ParPop() const {
		return *_p_pop;
	}

private:

	ABConvertor(const ABConvertor&);

	VALUE_MEMBER_REF

	parameters::OneDMInputSetParameter _param_input;
	parameters::OneDMParameter _param_onedm;
	ABScalarProduct _scalar_product;
	const parameters::PopulistSpecificParameter* _p_specific;
	const parameters::PopulationParameter* _p_pop;
	const Number* _p_n_bins;
	const Potential* _p_delta_v;
};
} /* namespace populist */
} /* namespace MPILib */

#endif //include guard MPILIB_POPULIST_ABCONVERTOR_HPP_
