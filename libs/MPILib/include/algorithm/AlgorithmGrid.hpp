// Copyright (c) 2005 - 2010 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef MPILIB_ALGORITHMS_ALGORITHMGRID_HPP_
#define MPILIB_ALGORITHMS_ALGORITHMGRID_HPP_

#include <vector>
#include <valarray>
#include <MPILib/include/BasicTypes.hpp>

namespace MPILib {
namespace algorithm {

//! AlgorithmGrid
class AlgorithmGrid {
public:

	/**
	 * Create the state for a AlgorithmGrid with a maximum number of elements
	 */
	AlgorithmGrid(Number);

	/**
	 * Create an AlgorithmGrid with just a state (usually a single number)
	 * @param The State of the algorithm
	 */
	AlgorithmGrid(const std::vector<double>&);

	/**
	 * Construct an AlgorithmGrid from
	 * @param a state
	 * @param an interpretation
	 * @post the size of the state and interpretation array needs to be the same
	 */
	AlgorithmGrid(const std::vector<double>&, const std::vector<double>&);

	/**
	 * Assignment operator
	 * @param The assigned AlgorithmGrid
	 * @return this
	 */
	AlgorithmGrid& operator=(const AlgorithmGrid&);

	/**
	 * Getter for the state vector
	 * @return the state vector
	 */
	std::vector<double> toStateVector() const;

	/**
	 * Getter for the interpretation vector
	 * @return the interpretation vector
	 */
	std::vector<double> toInterpretationVector() const;

private:

	template<class WeightValue> friend class AlgorithmInterface;

	/**
	 * Helper function to convert vector to valarray
	 * @param vector to be converted
	 * @return a valarray
	 */
	template<class Value>
	std::valarray<Value> toValarray(const std::vector<double>& vector) const;

	/**
	 * Helper function to convert valarray to vector
	 * @param array the valarray to be converted
	 * @param number_to_be_copied The number of elements copied
	 * @return a vector
	 */
	template<class Value>
	std::vector<Value> toVector(const std::valarray<Value>& array,
			Number number_to_be_copied) const;

	/**
	 * Getter for the state array
	 * @return the state array
	 */
	std::valarray<double>& getArrayState();
	/**
	 * Getter for the array interpretation
	 * @return the interpretation array
	 */
	std::valarray<double>& getArrayInterpretation();

	/**
	 * Getter for the number of states stored
	 * @return number of elements stored
	 */
	Number& getStateSize();
	/**
	 * const getter for the number of states stored
	 * @return number of elements stored
	 */
	Number getStateSize() const;

	/**
	 * Resize the arrays of elements
	 * @param The new size of the arrays
	 */
	void resize(Number);

	//! allow iteration over internal values of the state
	const double* begin_state() const;
	const double* end_state() const;

	const double* begin_interpretation() const;
	const double* end_interpretation() const;

	Number _numberState; // the array_state is sometimes larger than the actual state
	std::valarray<double> _arrayState;
	std::valarray<double> _arrayInterpretation;
};

}
} // end of namespace

#endif //MPILIB_ALGORITHMS_ALGORITHMGRID_HPP_ include guard
