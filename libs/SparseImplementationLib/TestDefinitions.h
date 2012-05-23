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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NETLIB_TESTDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_TESTDEFINITIONS_INCLUDE_GUAR

#include <string>

using std::string;

namespace SparseImplementationLib
{

	const string STR_IMPLEMENTATIONSTREAMING_TEST
					(
						"test_implementation.imp"
					);
	const string STR_IMPLEMENTATIONSTREAMING_TEST_COPY
					(
						"test_implementation_copy.imp"
					);
	const string STR_NODESTREAMING_TEST
					(
						"test_node.imp"
					);
	const string STR_TEST_DIR
					(
						"test/"
					);

	const string STR_IMPLEMENTATION_EXAMPLE
					(
						"SparseImplementationExample.collection"
					);

	const double NO_WEIGHT    =  0;
	const double F_WEIGHT_0_6 = -6;
	const double F_WEIGHT_0_7 = -7;
	const double F_WEIGHT_0_8 = -8;
	const double F_WEIGHT_0_9 = -9;
	const double F_WEIGHT_0_10 = -10;
	const double F_WEIGHT_1_6 = 101;
	const double F_WEIGHT_2_6 = 102;
	const double F_WEIGHT_3_6 = 103;
	const double F_WEIGHT_3_7 = 104;
	const double F_WEIGHT_3_8 = 105;
	const double F_WEIGHT_4_8 = 106;
	const double F_WEIGHT_5_8 = 107;
	const double F_WEIGHT_6_9 = 108;
	const double F_WEIGHT_7_9 = 109;
	const double F_WEIGHT_7_10 = 110;
	const double F_WEIGHT_8_10 = 111;

	const int NUMBER_OF_DYNAMIC_NODES = 10;



	template <class Implementation>
	inline bool InsertTestWeightsWithoutThreshold(Implementation& implementation)
	{ 
		return 
		( 
		    implementation.InsertWeight(NodeId(6),NodeId(1),F_WEIGHT_1_6) &&  
		    implementation.InsertWeight(NodeId(6),NodeId(2),F_WEIGHT_2_6) &&  
			implementation.InsertWeight(NodeId(6),NodeId(3),F_WEIGHT_3_6) &&  
			implementation.InsertWeight(NodeId(7),NodeId(3),F_WEIGHT_3_7) &&  
			implementation.InsertWeight(NodeId(8),NodeId(3),F_WEIGHT_3_8) &&  
			implementation.InsertWeight(NodeId(8),NodeId(4),F_WEIGHT_4_8) &&  
			implementation.InsertWeight(NodeId(8),NodeId(5),F_WEIGHT_5_8) &&  
			implementation.InsertWeight(NodeId(9),NodeId(6),F_WEIGHT_6_9) &&  
			implementation.InsertWeight(NodeId(9),NodeId(7),F_WEIGHT_7_9) &&  
			implementation.InsertWeight(NodeId(10),NodeId(7),F_WEIGHT_7_10) && 
			implementation.InsertWeight(NodeId(10),NodeId(8),F_WEIGHT_8_10)
		);	
	}

	template <class Implementation>
	inline bool InsertTestWeights(Implementation& implementation)
	{ 

		return ( 
					
					InsertTestWeightsWithoutThreshold(implementation) &&
					(
						implementation.InsertWeight(NodeId(6),NodeId(0),F_WEIGHT_0_6)   && 
						implementation.InsertWeight(NodeId(7),NodeId(0),F_WEIGHT_0_7)   &&
						implementation.InsertWeight(NodeId(8),NodeId(0),F_WEIGHT_0_8)   &&
						implementation.InsertWeight(NodeId(9),NodeId(0),F_WEIGHT_0_9)   &&
						implementation.InsertWeight(NodeId(10),NodeId(0),F_WEIGHT_0_10) 
					)
				);
	}

} // end of Netlib

#endif // include guard
