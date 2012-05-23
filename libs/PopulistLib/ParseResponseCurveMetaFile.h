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
#ifndef _CODE_LIBS_POPULISTLIB_PARSERESPONSECURVEMETAFILE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_PARSERESPONSECURVEMETAFILE_INCLUDE_GUARD

#include <string>
#include "../UtilLib/UtilLib.h"
#include "../DynamicLib/DynamicLib.h"
#include "OrnsteinUhlenbeckParameter.h"
#include "PopulistSpecificParameter.h"

using DynamicLib::AbstractReportHandler;
using DynamicLib::SimulationRunParameter;
using UtilLib::SequenceIteratorIterator;
using std::string;

namespace PopulistLib {

	class ParseResponseCurveMetaFile {
	public:

		ParseResponseCurveMetaFile(const string&);

		bool	ParsePopulationParameter	(istream&);
		
		bool	ParseSimulationParameter	(istream&);

		PopulationParameter					ParPop() const;

		SimulationRunParameter				
			ParSim
			(
				const AbstractReportHandler&,
				const string&
			) const;

		ParameterScan&						ParScan() const;

		PopulistSpecificParameter			ParSpec() const;

	private:

		bool	Parse						();
		void	WriteDefaultFile			() const;

		string										_path_name;
		PopulationParameter							_par_pop;
		boost::shared_ptr<ParameterScan>			_p_scan;
		boost::shared_ptr<SimulationRunParameter>	_p_par_run;
		istream*									_p_str_current;

		Time								_t_start;
		Time								_t_step;
		Time								_t_end;
		Time								_t_report;
		Time								_t_net;
		Time								_t_update;

	};
}

#endif // include guarf
