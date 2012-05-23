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
#include <boost/tokenizer.hpp>
#include "WorkFlow.h"
#include "WorkFlowDefinitions.h"
#include "PopulistException.h"

using namespace PopulistLib;

WorkFlow::WorkFlow
(
	const string& directory,
	const string& handler_name):
_directory(directory),
_handler_name(handler_name),
_handler(directory + string("/") + handler_name,false,true) // do not visualize, but write to file
{
}

WorkFlow::~WorkFlow()
{
}

SimulationRunParameter WorkFlow::WORKFLOW_SimulationRunParameter() const
{
	string runparameter_name = this->_directory + string("/") + this->WorkFlowName() + string(".runpar");
	string log_name = this->_directory + string("/") + this->WorkFlowName() + string(".log");

	ifstream str_runpar(runparameter_name.c_str());
	if (str_runpar) {
		SimulationRunParameter par(_handler,str_runpar);
		return par;
	}
	else {

		ofstream str(runparameter_name.c_str());
		
		if (! str)
			throw PopulistException("Cannot open SimulationRunParameter file for writing");

		SimulationRunParameter 
			par_ret
			(
				_handler,
				WORKFLOW_DEFAULT_MAXITER,
				WORKFLOW_DEFAULT_TIME_START,
				WORKFLOW_DEFAULT_TIME_END,
				WORKFLOW_DEFAULT_TIME_REPORT,
				WORKFLOW_DEFAULT_TIME_UPDATE,
				WORKFLOW_DEFAULT_TIME_NETWORKSTEP,
				log_name
			);

		par_ret.ToStream(str);
		return  par_ret;
	}
}

bool WorkFlow::WORKFLOW_Execute() const
{
	SimulationRunParameter par_run = this->WORKFLOW_SimulationRunParameter();
	return true;
}

string WorkFlow::WorkFlowName() const
{
	typedef boost::tokenizer<boost::char_separator<char> >  tokenizer;
	boost::char_separator<char> sep(".");
	tokenizer tokens(_handler_name, sep);
	return *(tokens.begin());
}
