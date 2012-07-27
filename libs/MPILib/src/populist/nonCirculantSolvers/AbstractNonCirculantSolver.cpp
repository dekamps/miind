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
#include <MPILib/include/populist/nonCirculantSolvers/AbstractNonCirculantSolver.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

namespace MPILib {
namespace populist {
namespace nonCirculantSolvers {

AbstractNonCirculantSolver::AbstractNonCirculantSolver(CirculantMode mode) :
		_array_factor(0), _epsilon(EPS_J_CIRC_MAX), _mode(mode) {
}

bool AbstractNonCirculantSolver::Configure(std::valarray<double>& array_state,
		const parameters::InputParameterSet& input_set, double epsilon) {
	if (epsilon == 0)
		_epsilon = EPS_J_CIRC_MAX;
	else
		_epsilon = epsilon;

	_p_array_state = &array_state;

	_p_input_set = &input_set;

	return true;
}

void AbstractNonCirculantSolver::InitializeArrayFactor(Time tau,
		Number n_non_circulant) {
	assert(_epsilon > 0);
	if (_epsilon < EPS_J_CIRC_MAX)
		_epsilon = EPS_J_CIRC_MAX;

	if (n_non_circulant > _array_factor.size())
		_array_factor.resize(n_non_circulant);

	_array_factor = 0.0;
	_array_factor[0] = exp(-tau);
	for (int i = 1; i < static_cast<int>(n_non_circulant); i++) {
		// Let some precision criterion determine where this breaks off
		// and store the break off value
		_array_factor[i] = tau * _array_factor[i - 1] / i;
		if (_array_factor[i] < _epsilon) {
			_j_circ_max = i;
			return;
		}
	}
	_j_circ_max = n_non_circulant;
}
} /* namespace nonCirculantSolvers */
} /* namespace populist */
} /* namespace MPILib */
