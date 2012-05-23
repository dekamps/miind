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
#ifndef _CODE_APPS_UNITDYNAMICLIB_SINGLEPOPULATIONNETWORKFIXTURECODE_INCLUDE_GUARD
#define _CODE_APPS_UNITDYNAMICLIB_SINGLEPOPULATIONNETWORKFIXTURECODE_INCLUDE_GUARD

#include "SinglePopulationNetworkFixture.h"
#include "LocalDefinitions.h"

namespace UnitDynamicLib {

	const long	DEF_RUN_MAXITER		= 1000000;					// maximum number of iterations
	const Time	DEF_RUN_T_START		= 0;						// start time of simulation
	const Time	DEF_RUN_T_END		= 0.5;						// end time of simulation
	const Time	DEF_RUN_T_REPORT	= 1e-4;						// report time
	const Time	DEF_RUN_T_UPDATE	= 1e-4;						// update time
	const Time  DEF_RUN_T_NET		= 1e-5;						// network step time

	template <class Weight>
	SinglePopulationNetworkFixture<Weight>::SinglePopulationNetworkFixture
	(
	):_par_run(DEFAULT_RUN_PARAMETER)
	{
	}

	template <class Weight>
	void SinglePopulationNetworkFixture<Weight>::SetOutputFileNames
	(
		const string& name_results,
		const string& name_log
	)
	{
		AsciiReportHandler handler("name_results");
		_par_run = 
			SimulationRunParameter
			(
				handler,
				DEF_RUN_MAXITER,					// maximum number of iterations
				DEF_RUN_T_START,					// start time of simulation
				DEF_RUN_T_END,						// end time of simulation
				DEF_RUN_T_REPORT,					// report time
				DEF_RUN_T_UPDATE,					// update time
				DEF_RUN_T_NET,						// network step time
				name_log
			);
	}
}
#endif // include guard