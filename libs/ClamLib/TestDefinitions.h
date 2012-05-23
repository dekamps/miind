// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef _CODE_LIBS_CLAMLIB_TESTDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_TESTDEFINITIONS_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "../StructnetLib/StructnetLib.h"
#include "BasicDefinitions.h"

using StructnetLib::LayerDescription;
using DynamicLib::Time;
using DynamicLib::WilsonCowanParameter;

namespace ClamLib {

			// categorization value which keeps the squashing function
			// in a linear range
			const DynamicLib::Rate MAX_LINEAR_STATIC_RATE = 0.05;

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
				24, // nr x pixels
				24, // nr y pixels
				4,  // nr orientations
				1,  // size of receptive field in x
				1,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription LAYER_1 = 
			{ 
				23,  // nr x pixels
				23,  // nr y pixels
				1,  // nr orientations
				2,  // size of receptive field in x
				2,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription LAYER_2 = 
			{  
				21,  // nr x pixels
			    21,  // nr y pixels
				1,  // nr orientations
				3,  // size of receptive field in x
				3,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription LAYER_3 = 
			{  
				17,  // nr x pixels
				17,  // nr y pixels
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
				17,  // size of receptive field in x
				17,  // size of receptive field in y
				1,  // nr x skips
				1   // nr y skips
			}; 

		const LayerDescription SMALL_LAYER_0 =
		{
				2,
				2,
				2,
				1,
				1,
				1
		};

		const LayerDescription SMALL_LAYER_1 =
		{
				2,
				2,
				1,
				2,
				2,
				1,
				1
		};

		const LayerDescription SMALL_LAYER_2 =
		{
				1,
				1,
				2,
				2,
				2,
				1,
				1
		};
		const WilsonCowanParameter PARAM_EXC(20e-3,1,1);
		const WilsonCowanParameter PARAM_INH(10e-3,1,1);

			const Number NR_JOCN_OUTPUTS = 4;

			const int TR_JOCN = 11;

			const double JOCN_ENERGY = 1.0e-5;

			const DynamicLib::Time FUNCTOR_TEST_TIME = 0.3;

			const double REVERSE_SCALE = 20.0;


			const string NAME_FFD_DEVELOP("developffd.net");

			const string NAME_REV_DEVELOP("developrev.net");

			const string NAME_JOCN_FORWARD_ROOT("jocn_fwd.root");

			const string NAME_JOCN_FUNCTOR_ROOT("jocn_functor.root");

			const string NAME_JOCN_FFDFDBCK("jocn_ffdfdbck.root");

			const string NAME_JOCN_DISINHIBITION("jocn_disinhibition.root");

			const string TEST_PATH("test/");


}

#endif // include guard
