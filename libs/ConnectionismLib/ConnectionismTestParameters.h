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

#ifndef _CODE_LIBS_CONNECTIONISM_CONNECTIONISMTESTPARAMETERS_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISM_CONNECTIONISMTESTPARAMETERS_INCLUDE_GUARD


#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "TrainingParameter.h"

namespace ConnectionismLib
{
	const Number NR_XOR_TRAINING_STEPS = 2500;
	const double XOR_TEST_ENERGY       = 0.01;

	// TrainingParameter for the XOR test
	const TrainingParameter XOR_PARAMETER
		(
			0.1, // stepsize
			1.0,    // sigma of weight initialization
			0,    // bias of weight initialization
			1,    // number of steps per training (for batch)
			true, // thresholds must be trained (essential for XOR)
			0,    // threshold default value
		        0,  // Backprop momentum term
			0,    // default seed
			1 
		);  // initailize only once


	const TrainingParameter XOR_SINGLE_STEP_PARAMETER
		(
			1,  // stepsize
			0,    // sigma of weight initialization
			1,    // bias of weight initialization
			1,    // number of steps per training (for batch)
			true, // thresholds must be trained (essential for XOR)
			0,    // threshold default value
			0,  // Backprop momentum term
			0,    // default seed
			1 
		);  // initailize only once


	const TrainingParameter HEBBIAN_PARAMETER
		(
			1,  // stepsize
			0,    // sigma of weight initialization
			0,    // bias of weight initialization
			1,    // number of steps per training (for batch)
			true, // thresholds must be trained (essential for XOR)
			0,    // threshold default value
			0.9,  // Backprop momentum term
			0,    // default seed
			1 
		);  // initailize only once


	// Check backprop for a single step in the XOR problem

	const double DELTA_1 = -0.38800382879;
	const double DELTA_2 = -0.15257236576;
	const double EPSILON_XOR = 1e-8;

} // end of Connectionism


#endif // include guard
