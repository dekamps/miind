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
#ifndef _CODE_LIBS_POPULILISTLIB_LOCAL_DEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULILISTLIB_LOCAL_DEFINITIONS_INCLUDE_GUARD

#include <string>
#include "../UtilLib/UtilLib.h"
#include "BasicDefinitions.h"

using UtilLib::Index;
using UtilLib::Number;
using std::string;

#ifdef _INVESTIGATE_ALGORITHM
#define VALUE_ARG vec_value,
#define VALUE_MEMBER_ARG _vec_value,
#define VALUE_REF vector<ReportValue>&,
#define VALUE_REF_INIT vector<ReportValue>& vec_value,
#define VALUE_MEMBER vector<ReportValue> _vec_value;
#define VALUE_MEMBER_REF vector<ReportValue>& _vec_value;
#define VALUE_MEMBER_INIT _vec_value(vec_value),
#define VALUE_RETURN		virtual vector<ReportValue> GetValues() const { return _vec_value; }
#else
#define VALUE_ARG
#define VALUE_MEMBER_ARG 
#define VALUE_REF
#define VALUE_REF_INIT
#define VALUE_MEMBER 
#define VALUE_MEMBER_REF 
#define VALUE_MEMBER_INIT
#define VALUE_RETURN
#endif


namespace PopulistLib {

	const Number NONCIRC_LIMIT = 5;

	const Number CIRCULANT_POLY_DEGREE = 4;

	const Index CIRCULANT_POLY_JMAX = 7;

	const double DOUBLE_POPULATION_STEP_SIZE_FRACTION = 0.02;

	const int MAX_V_ARRAY = 100000; // should be more than sufficient

	const int MAXIMUM_NUMBER_GAMMAZ_VALUES = 500;

	const int MAXIMUM_NUMBER_CIRCULANT_BINS     = 100000;
	const int MAXIMUM_NUMBER_NON_CIRCULANT_BINS = 100000;

	const int N_CIRC_DIFF = 50;

	const Number NUMBER_INTEGRATION_WORKSPACE = 10000;

	const Number N_CIRC_MAX_CHEB = 50;

	const Number N_NON_CIRC_MAX_CHEB = 70;

	const Time T_CHEB_MIN = 0.0;

	const Time T_CHEB_MAX = 0.3;

	const double CHEB_PRECISION = 1e-8;

	const double ALPHA_LIMIT = 1e-6;

	//TODO: restore
	const double EPSILON_INTEGRALRATE = 1e-4;

	//TODO: restore
	const double RELATIVE_LEAKAGE_PRECISION = 1e-4;//1e-10;

	const double FIT_LOWER_BOUND = 0.05;

	const string 
		STR_BINS_MUST_BE_ADDED
		(
			"You have specified a finite time constant, but no bins to be added. This is inconsistent"
		);

	const string STR_MEMBRANE_ZERO
			(
				"You have taken the membrane time constant equal to zero. Evolution can not proceed"
			);

	const string STR_UNKNOWN_REBINNER
			(
				"You have tried to invoke an unknown rebinning algorithm"
			);

	const string STR_LOOKUP_DISABELED
			(
				"The lookup table mode is disabeled. The VMatrix code now only serves in the FULL_CALCULATION mode as a check on the regular method.\nYou should not use VMatrix in your code"
			);

	const string STR_ROUND_OFF_ERROR_TO_LOG
			(
				"Can not require the precision specified for the rate integration"
			);

	const string WORKSPACE_EXCESSION
			(
				"Interpolation workspace exceeded. Recompile if necessary"
			);

	const string STR_UNKNOWN_MODE
			(
				"This is an unknown population mode"
			);

	const string STR_INCORRECT_SPECIFIC_PARAMETER
			(
				"You tried to feed a specific parameter that is not a PopulistSpecificParameter"
			);

	const string STR_SPECIFIC_PARAMETER_FORGOTTEN
			(
				"Did you forget to initialize a PopulistSpecificParameter ?"
			);

	const string STR_RATE_EXC_NEG
			(
				"A negative rate was calculated for the excitatory input population. Use single population or increase the number of bins"
			);

	const string STR_RATE_INH_NEG
			(
				"A negative rate was calculated for the inhibitory input population. Use single population or increase the number of bins"
			);
	const double EPSILON = 1e-4;


	const string STR_VLOOKUP_H
			(
				"PopulistVLookUp.h"
			);

	const string STR_VLOOKUP_CPP
			(
				"PopulistVLookUp.cpp"
			);

	const string 
		STR_OU_UNEXPECTED
	    (
			"Unexpected OrnsteinUhlenbeckParameter tag"
		);

	const string
		STR_QIFP_UNEXPECTED
		(
			"Unexpected QIFParameter tag"
		);

	const string
		STR_POPALGORITHM_TAG
		(
			"<PopulationAlgorithm>"
		);

	const string
		STR_POPULISTSPECIFIC_TAG
		(
			"<PopulationSpecific>"
		);

	const double F_ABS_REL_BOUNDARY = 1;                   // if upper limit > F_ABS_REL_BOUNDARY, use relative error
	const double EPSILON_RESPONSE_FUNCTION         = 1e-6; // absolute/relative error on spike response function
	const double MAXIMUM_INVERSE_RESPONSE_FUNCTION = 5;    // if upper limit is above this value, it might just as well be infinite

	const int OU_STATE_DIMENSION     = 1;
	const int OU_PARAMETER_DIMENSION = 1;

	const double OU_ABSOLUTE_PRECISION = 1e-5;
	const double OU_RELATIVE_PRECISION = 0;

	const string STRING_NUMBER_ITERATIONS_EXCEEDED
					(
						"The predetermined number of iterations is exceeded in Evolve()"
					);

}

#endif // include guard
