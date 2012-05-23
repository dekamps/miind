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
#ifndef _CODE_LIBS_STRUCTNET_BASICDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_STRUCTNET_BASICDEFINITIONS_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <string>
#include "../NetLib/NetLib.h"
#include "../UtilLib/UtilLib.h"
#include "SparseLayeredNet.h"


using std::string;
using ConnectionismLib::BackpropTrainingVector;
using ConnectionismLib::Hebbian;
using ConnectionismLib::TrainingParameter;
using UtilLib::Number;

namespace StructnetLib
{

		const string NETLIB_TEST_DIR("test/");               // test directory, containing files necessary for NetLibtest
		const string NETLIB_TEST_FWD_NET_FILE("net.txt");    // name of forward network test file
		const string NET_LIB_TEST_PATTERN_BASE_NAME("pat_"); // All files have this base name
		const string OPATTERN_TEST_FILE("opattern.pat");     // test file for streaming of oriented patterns
		const string PATTERN_EXTENSION(".txt");

		const Number NR_PATTERN_FILES     = 16;              // There are 36 pattern files
		const Number NR_OUTPUT_CATEGORIES = 4;               // The first 9 patterns are translated version of the first pattern,
	                                                         // hence there are 4 output categories

		const Number OP_N_X = 3;
		const Number OP_N_Y = 3;
		const Number OP_N_Z = 3;

		const TrainingParameter NETLIB_TRAINING_PARAMETER
			(  
				.05,       // step size
				1,        // sigma
				0,        // bias
				1,        // nr_steps
				false,    // do not train thresholds
				0.9,      // momentum
				0,        // no threshold value
				282333,   // some seed
				1         // initialize once 
			); 
		const double      NETLIB_TEST_ENERGY   = 0.1;


		// Dimensions of a testing network of five layers:

		const LayerDescription LAYER_0 = 
			{ 
				12, // nr x pixels
				12, // nr y pixels
				4,  // nr orientations
				1,  // size of receptive field in x
				1,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription LAYER_1 = 
			{ 
				11,  // nr x pixels
				11,  // nr y pixels
				1,  // nr orientations
				2,  // size of receptive field in x
				2,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription LAYER_2 = 
			{  
				9,  // nr x pixels
			    9,  // nr y pixels
				1,  // nr orientations
				3,  // size of receptive field in x
				3,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription LAYER_3 = 
			{  
				5,  // nr x pixels
				5,  // nr y pixels
				1,  // nr orientations
				5,  // size of receptive field in x
				5,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 


		const LayerDescription LAYER_4 = 
			{  
				1,  // nr x pixels
				1,  // nr y pixels
				4,  // nr orientations
				5,  // size of receptive field in x
				5,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		// Dimensions of a testing network of three layers:

		const LayerDescription MINI_0 = 
			{ 
				3, // nr x pixels
				3, // nr y pixels
				2,  // nr orientations
				1,  // size of receptive field in x
				1,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription MINI_1 = 
			{ 
				2,  // nr x pixels
				2,  // nr y pixels
				1,  // nr orientations
				2,  // size of receptive field in x
				2,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription MINI_2 = 
			{  
				1,  // nr x pixels
			    1,  // nr y pixels
				3,  // nr orientations
				3,  // size of receptive field in x
				3,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

	

} // end of Strucnet


#endif // include guard
