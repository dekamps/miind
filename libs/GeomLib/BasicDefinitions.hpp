// Copyright (c) 2005 - 2014 Marc de Kamps
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
#ifndef _CODE_LIBS_GEOMLIB_BASICDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_BASICDEFINITIONS_INCLUDE_GUARD

#include <MPILib/include/BasicDefinitions.hpp>
//#include "../UtilLib/UtilLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"
using UtilLib::Number;

//! GeomLib

namespace GeomLib
{
	//! typedef floating point number between or equal to 0 and 1
	typedef double Fraction;

	//! Standard start time of simulation
	const MPILib::Time T_START = 0.0;

	//! Default time step
	const MPILib::Time T_STEP = 1e-6;

	//! Default precision for running numerical integrations
	const NumtoolsLib::Precision PRECISION(1e-7,0.0);

	//! Maximum number of iterations before DynamicNetwork reports a failure
	const Number MAXITERATION = 10000000;

	//! Maximum number of terms that the DiffusionMasterEquation should take into account
	const Number N_DIFF_MAX  = 20;

	//! Batch times for probability queues
	const MPILib::Time TIME_QUEUE_BATCH = 1e-6;

} // end of GeomLib

#endif // include guard
