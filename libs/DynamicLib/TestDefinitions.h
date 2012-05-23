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

#ifndef _CODE_LIBS_DYNAMICLIB_TESTDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_TESTDEFINITIONS_INCLUDE_GUARD

#include "LocalDefinitions.h"
#include "SimulationRunParameter.h"
#include "RootReportHandler.h"

namespace DynamicLib
{
	// Just a large number
	const Number NUMBER_OF_RESERVED_NODES = 1000;

	const string 
		STRING_TEST_DIR
		(
			"test/"
		);
	const string 
		STRING_WILSONCOWAN_FILE
		(
				"WilsonCowan.net"
		);

	const string 
		STRING_WILSONCOWAN_ROOTFILE
		(
			"WilsonCowan.root"
		);

	const string 
		STRING_HIGHTHROUGHPUT_ROOTFILE
		(
			"HighThroughput.root"
		);

	const string 
		STRING_NODESTATE_STREAMING_TEST
		(
			"NodeState.state"
		);

	const string 
		STRING_GRID_STREAMING_TEST
		(
			"AlgorithmGrid.grid"
		);

	// CowanWilsonAlgorithm Definitions
	const Rate   F_MAX    = 500.0;
	const double F_NOISE  =  1.0;
	const double AFFERENT = 1.0;
	const Rate F_RESULT   = F_MAX/( 1 + exp(-1*AFFERENT));
	const double EPSILON  = 1e-6;
	const Time   TIME_MEMBRANE = 5e-3;

	//CowanWilsonNetwork deinitions

	const double ALPHA = 1;
	const double BETA  = -1.1;
	const double GAMMA = 2;
	const double DELTA = -2;
	const double ETA   = -1.5;

	const Time   TAU_EXCITATORY  = 10e-3;
	const Time   TAU_INHIBITORY  = 5e-3;
	const Time   T_START         = 0;
	const Time   T_END           = 50e-3;
	const Time   T_STEP          = 1e-3; // ms

	const vector<double>  WILSON_COWAN_GRID_VECTOR(1,0);
	const AlgorithmGrid   
		WILSON_COWAN_GRID
		(
			WILSON_COWAN_GRID_VECTOR, 
			WILSON_COWAN_GRID_VECTOR
		);

	// define a handler to store the simulation results
	const RootReportHandler 
		WILSONCOWAN_HANDLER
		(
			"test/wilsonresponse.root",	// file where the simulation results are written
			false,						// do not display on screen
			false						// only rate diagrams
		);

	const SimulationRunParameter
		PAR_WILSONCOWAN
		(
			WILSONCOWAN_HANDLER,		// the handler object
			1000000,					// maximum number of iterations
			0,							// start time of simulation
			0.5,						// end time of simulation
			1e-4,						// report time
			1e-4,						// update time
			1e-5,						// network step time
			"test/wilsonresponse.log"   // log file name
		);

	const string 
		STRING_ASCIIREPORTNAME
		(
			"WilsonCowan3NodesReport"
		);

	const string 
		STRING_OPEN_REPORT_FAILED
		(
			"Couldn't open report file"
		);
} 

#endif // include guard
