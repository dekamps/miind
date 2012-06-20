// Copyright (c) 2005 - 2008 Marc de Kamps
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

#include <MPILib/include/algorithm/AlgorithmGrid.hpp>
#include <cassert>
#include <functional>

namespace MPILib {
namespace algorithm {

AlgorithmGrid::AlgorithmGrid(Number number_of_elements) :
		_numberState(number_of_elements), _arrayState(0.0,
				number_of_elements), _arrayInterpretation(0.0,
				number_of_elements) {
}

AlgorithmGrid::AlgorithmGrid(const std::vector<double>& array_state) :
		_numberState(static_cast<Number>(array_state.size())), _arrayState(
				toValarray<double>(array_state)), _arrayInterpretation(
				std::valarray<double>(0.0, array_state.size())) {
}

AlgorithmGrid::AlgorithmGrid(const std::vector<double>& array_state,
		const std::vector<double>& array_interpretation) :
		_numberState(static_cast<Number>(array_state.size())), _arrayState(
				toValarray<double>(array_state)), _arrayInterpretation(
				toValarray<double>(array_interpretation)) {
	assert( _arrayState.size() == _arrayInterpretation.size());
}

AlgorithmGrid& AlgorithmGrid::operator=(const AlgorithmGrid& rhs) {
	if (&rhs == this)
		return *this;
	else {
		// resize, because copying valarrays of different length is undefined
		_arrayState.resize(rhs._arrayState.size());
		_arrayInterpretation.resize(rhs._arrayInterpretation.size());

		_arrayState = rhs._arrayState;
		_arrayInterpretation = rhs._arrayInterpretation;
		_numberState = rhs._numberState;
		return *this;
	}
}

std::vector<double> AlgorithmGrid::toStateVector() const {
	return toVector(_arrayState, _numberState);
}


std::vector<double> AlgorithmGrid::toInterpretationVector() const {
	return toVector(_arrayInterpretation, _numberState);
}

template<class Value>
std::valarray<Value> AlgorithmGrid::toValarray(
		const std::vector<double>& vector) const {
	return std::valarray<Value>(&vector[0], vector.size());
}

template<class Value>
std::vector<Value> AlgorithmGrid::toVector(const std::valarray<Value>& array,
		Number number_to_be_copied) const {
	auto& array_test = const_cast<std::valarray<Value>&>(array);
	auto p_begin = &array_test[0];
	auto p_end = p_begin + number_to_be_copied;

	std::vector<Value> vector_return(0);
	std::copy(p_begin, p_end, back_inserter(vector_return));

	return vector_return;
}

std::valarray<double>& AlgorithmGrid::getArrayState() {
	return _arrayState;
}

std::valarray<double>& AlgorithmGrid::getArrayInterpretation() {
	return _arrayInterpretation;
}

Number& AlgorithmGrid::getStateSize() {
	return _numberState;
}

Number AlgorithmGrid::getStateSize() const {
	return _numberState;
}


void AlgorithmGrid::resize(Number number_of_new_bins) {
	_arrayState.resize(number_of_new_bins);
	_arrayInterpretation.resize(number_of_new_bins);
}


const double* AlgorithmGrid::begin_state() const {

	auto& ref_state = const_cast<std::valarray<double>&>(_arrayState);
	const double* p_begin = &ref_state[0];
	return p_begin;
}

const double* AlgorithmGrid::end_state() const {
	auto& ref_state = const_cast<std::valarray<double>&>(_arrayState);
	const double* p_end = &ref_state[_numberState];
	return p_end;
}

const double* AlgorithmGrid::begin_interpretation() const {

	auto& ref_state = const_cast<std::valarray<double>&>(_arrayInterpretation);
	const double* p_begin = &ref_state[0];
	return p_begin;
}

const double* AlgorithmGrid::end_interpretation() const {
	auto& ref_state = const_cast<std::valarray<double>&>(_arrayInterpretation);
	const double* p_end = &ref_state[_numberState];
	return p_end;
}





}
} //end namespace MPILib
