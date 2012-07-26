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
#include <MPILib/include/populist/nonCirculantSolvers/MatrixNonCirculant.hpp>
#include <MPILib/include/utilities/Exception.hpp>


namespace MPILib {
namespace populist {
namespace nonCirculantSolvers {


MatrixNonCirculant::MatrixNonCirculant():
AbstractNonCirculantSolver(INTEGER)
{
}

MatrixNonCirculant::~MatrixNonCirculant()
{
}

MatrixNonCirculant* MatrixNonCirculant::Clone() const
{
	return new MatrixNonCirculant;
}

void MatrixNonCirculant::ExecuteExcitatory
(
	Number n_bins,
	Time   tau
)
{
	// This is the most straightforward version of the algorithm: simply create a single
	// row that contains exp Lt.
	// Expectation is that this will be slower than the standard NonCirculantSolver,
	// but that it is independent of the number of input populations to a first approximation,
	// because the time to set up the matrix row is shorter than the time to carry out the matrix 
	// multiplication.

	this->InitializeArrayFactor(tau,n_bins);


	int H = static_cast<int>(_p_input_set->_H_exc);
	_matrix_row = 0.0;
	int i,j;
	for (i = static_cast<int>(n_bins) - 1,  j = 0; i >=0; i-=H, j++)
		_matrix_row[i] = _array_factor[j];

	std::valarray<Potential>& array_state = *_p_array_state;

	for( int i = n_bins - 1; i >= 0; i-- )
	{
		// prevent the overwrite so that the matrix manipulation can be done in
		// the case i == j first
		array_state[i] = _matrix_row[n_bins - 1]*array_state[i];

		for( int j = 0; j < i; j++ )
			array_state[i] += _matrix_row[n_bins-i-1+j]*array_state[j];
	}

}

void MatrixNonCirculant::ExecuteInhibitory
(
	Number n_bins,
	Time tau
)
{
	throw utilities::Exception("Not yet implemented");
}

bool MatrixNonCirculant::Configure
(
	std::valarray<double>&		 array_state,
	const parameters::InputParameterSet& input_set,
	double
)
{
	// Normally this is done in the base class 
	_p_array_state = &array_state;
	_p_input_set = &input_set;

	// however we need an overloaded Configure for this reason:
	_matrix_row = array_state;
	return true;
}
} /* namespace nonCirculantSolvers */
} /* namespace populist */
} /* namespace MPILib */
