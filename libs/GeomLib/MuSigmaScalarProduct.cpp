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
#include "MuSigmaScalarProduct.hpp"
#include "ConnectionSquaredProduct.hpp"
#include <math.h>
#include <functional>
#include <numeric>

namespace GeomLib {

template <>
MPILib::Potential MuSigmaScalarProduct<double>::InnerProduct(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<double>& weightVector) const {

	return std::inner_product(nodeVector.begin(), nodeVector.end(),
			weightVector.begin(), 0.0);
}

template <>
MPILib::Potential MuSigmaScalarProduct<double>::InnerSquaredProduct(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<double>& weightVector) const {

	return 0;
}


template <>
MPILib::Potential MuSigmaScalarProduct<MPILib::DelayedConnection>::InnerProduct(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<MPILib::DelayedConnection>& weightVector) const {

	return std::inner_product(nodeVector.begin(), nodeVector.end(),
			weightVector.begin(), 0.0);
}


template <>
MPILib::Potential MuSigmaScalarProduct<MPILib::DelayedConnection>::InnerSquaredProduct(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<MPILib::DelayedConnection>& weightVector) const {

	return inner_product(nodeVector.begin(), nodeVector.end(),
			weightVector.begin(), 0.0, std::plus<double>(),
			ConnectionSquaredProduct());
}


template <>
MuSigma MuSigmaScalarProduct<MPILib::DelayedConnection>::Evaluate(const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<MPILib::DelayedConnection>& weightVector,
		MPILib::Time tau) const {
	MuSigma ret;

	ret._mu = tau * this->InnerProduct(nodeVector, weightVector);
	ret._sigma = sqrt(
			tau * this->InnerSquaredProduct(nodeVector, weightVector));

	return ret;
}

}

