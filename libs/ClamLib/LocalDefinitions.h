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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_CLAMLIB_LOCALDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_LOCALDEFINITIONS_INCLUDE_GUARD

#include <string>

using std::string;

namespace ClamLib {

	const double CIRCUIT_WEIGHT = 2.0;

	const double HOMEOSTATIC_SMOOTH_SCALE = 0.3;

	//! Wilson Cowan nodes have single double as state
	const int SEMI_SIGMOID_STATE_DIMENSION = 1;


	//! The parameter vector for Wilson Cowan integration has four elements
	const int SEMI_SIGMOID_PARAMETER_DIMENSION = 4;

	const double SS_ABSOLUTE_PRECISION = 1e-5;
	const double SS_RELATIVE_PRECISION = 0;


	const string STR_UNKNOWN_SQUASHING_FUNCTION("Don't know how to covert this squashing function into populations");

	const string STR_UNDEFINED_ALGORITHM("Algorithm pointer undefined");

	const string STR_TRAINING_FAILED("Couldn't train test network");

	const string STR_WEIGHTLIST_NON_EMPTY("This entry in weightlist was already taken");

	const string STR_NOTHING_TO_CONVERT("DynamicNetwork has 0 nodes");

	const string STR_EMPTY_NODELISTS("CircuitCreation has failed");

	const string STR_COMBINATION_NOT_IMPLEMENTED("This connection possibility isn't implemented");

	const string TAG_CIRCUIT_INFO("<CircuitInfo>");

	const string STR_SS_TAG("<SemiSigmoid>");

	typedef double Efficacy;
}

#endif // include guard