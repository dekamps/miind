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

#ifndef _CODE_LIBS_DYNAMICLIB_LOCALDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_LOCALDEFINITIONS_INCLUDE_GUARD

#include <string>

using std::string;

namespace DynamicLib
{
	typedef double Rate;
	typedef double Time;
	typedef Time   TimeStep;
	
	//! Rate Algorithm nodes have a single state
	const int RATE_STATE_DIMENSION = 1;

	//! Wilson Cowan nodes have single double as state
	const int WILSON_COWAN_STATE_DIMENSION = 1;


	//! The parameter vector for Wilson Cowan integration has four elements
	const int WILSON_COWAN_PARAMETER_DIMENSION = 4;

	const double WC_ABSOLUTE_PRECISION = 1e-5;
	const double WC_RELATIVE_PRECISION = 0;

	const int N_FRACT_PERCENTAGE_SMALL = 100;
	const int N_PERCENTAGE_SMALL       = 5;
	const int N_FRACT_PERCENTAGE_BIG   = 10;

	const int KEY_PRECISION = 8;

	const string STR_ASCIIHANDLER_EXCEPTION
					(
						"Could not open ascii handler file stream:"
					);

	const string STR_HANDLER_STALE
					(
						"This handler already has written reports"
					);

	const string STR_DYNAMICLIB_EXCEPTION
					(
						"Some DynamicLib exception occurred"
					);

	const string STR_STATE_CONFIGURATION_EXCEPTION
					(
						"There is a mismatch between the dimension of the State and the EvolutionAlgorithm"
					);

	const string STR_ROOTFILE_EXCEPTION
					(
						"Couldn't open root file"
					);

	const string STR_AE_TAG
					(
						"<AbstractAlgorithm>"
					);

	const string STR_AE_EXCEPTION
					(
						"Can't serialize an AbstractAlgorithm"
					);

	const string STR_NETWORKSTATE_TAG
					(
						"<NetworkState>"
					);

	const string STR_RA_TAG
					(
						"<RateAlgorithm>"
					);

	const string STR_NETWORKSTATE_ENDTAG
					(
						"</NetworkState>"
					);

	const string STRING_WC_TAG
					(
						"<WilsonCowanAlgorithm>"
					);
	const string STR_DYNAMICNETWORKIMPLEMENTATION_TAG
					(
						"<DynamicNetworkImplementation>"
					);
	const string STR_OPENNETWORKFILE_FAILED
					(
						"Could not open test file. Does test directory exist ?"
					);

	const string STR_DYNAMIC_NETWORK_TAG
					(
						"<DynamicNetwork>"
					);
	const string STR_NETWORK_CREATION_FAILED
					(
						"Creation of test dynamic network failed"
					);
	const string STR_EXCEPTION_CAUSE_UNKNOWN
					(
						"Unknow exception thrown in Evolve"
					);
	const string STR_NUMBER_ITERATIONS_EXCEEDED
					(
						"The predetermined number of iterations is exceeded in Evolve()"
					);
	const string STR_INCOMPATIBLE_TIMING_ERROR
					(
						"Node Evolve algorithm didn't reach specified end time"
					);
	const string STR_TIME
					(
						"<Time>"
					);
	const string STR_NODEID
					(
						"<NodeId>"
					);
	const string STR_REPORT
					(
						"<Report>"
					);

	const string STR_GRID_TAG
					(
						"<AlgorithmGrid>"
					);
	const string STR_GRID_PARSE_ERROR
					(
						"Error parsing AlgorithmGrid"
					);
	const string STR_STATE_PARSE_ERROR
					(
						"Error parsing NodeState"
					);
	const string STR_NODESTATE_TAG
					(
						"<NodeState>"
					);
	const string STR_DYN_TAG
					(
						"<DynamicNode>"
					);
	const string STR_WCP_TAG
					(
						"<WilsonCowanParameter>"
					);
	const string STRING_NODEVALUE
					(
						"<NodeValue>"
					);

	const string CANVAS_NAME
					(
						"Canvas"
					);

	const string CANVAS_TITLE
					(
						"Dynamic population overview"
					);

	const string STR_NETWORK_IMPLEMENTATION_REALLOC
					(
						"Network implementation will realloc. This invalidates the internal representation"
					);

	const string STR_ROOT_FILE_OPENED_FAILED
					(
						"Could not open ROOT file"
					);

	const string SIMRUNPAR_UNEXPECTED("Unexpected tag for SimulationRunParameter");

	const string OFFSET_ERROR("Cast to DynamicNode failed");

	const int CANVAS_X_DIMENSION = 800;
	const int CANVAS_Y_DIMENSION = 800;

} // end of DynamicLib

#endif // include guard

