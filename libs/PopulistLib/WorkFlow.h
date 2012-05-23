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
#ifndef _CODE_LIBS_POPULISTLIB_WORKFLOW_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_WORKFLOW_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "../UtilLib/UtilLib.h"
#include "../NetLib/NetLib.h"
#include "OrnsteinUhlenbeckAlgorithm.h"
#include "OrnsteinUhlenbeckParameter.h"

using DynamicLib::RootReportHandler;
using DynamicLib::SimulationRunParameter;
using NetLib::NodeId;

namespace PopulistLib {

	typedef Index WorkFlowId;

	//! A WorkFlow allows customization of a DynamicLib simulation. The customization is done by managing configuration files.
	//!
	//! Creating DynamicLib simulations is often a repetetive process. At the moment, a simulation is first typically programmed, that is, C++ or Python code
	//! is written to set up and run the simulation. In principle each run requires a new compilation, unless the devloper of the simulation creates their own 
	//! simulation files. WorkFlow classes are aimed at taking away this burden and at helping to create disk based representations of the simulation process.
	//! In the long run the aim is to be able to set up the simulations from configuration files alone. Current usages requires the following steps:
	//! <ul>
	//! <li> The user subclasses from WorkFlow.
	//! <li> The user overloades the functions starting with WORKFLOW_. In most cases the user most provide these program elements anyway and the WorkFlow is a suitable way of organising this code.</li>
	//! <li> Run the simulation. After the first run, there will be disk-based representations for each overloaded WORKFLOW method which can be modified in a text editor. From now on the simulation can be re-run without compilations.</li>
	//! </ul>

	class WorkFlow {
	public:

		WorkFlow
		(
			const string&,		/*!< Test directory name */
			const string&		/*!< Handler name */
		);

		virtual ~WorkFlow() = 0;

		WorkFlow(const WorkFlow&);

		WorkFlow& operator=(const WorkFlow&);

		virtual SimulationRunParameter WORKFLOW_SimulationRunParameter() const;

		virtual bool WORKFLOW_Execute() const;

	protected:

		//!	The WorkFlow name will be constructed from the handler name. If the name contains a period, such as in 'bla.root', the name will be 'bla', otherwise it will
		//! just be the handler name
		string WorkFlowName() const;

		string DirectoryName() const { return _directory; }

	private:

		const string								_directory;
		const string								_handler_name;
		RootReportHandler							_handler;

	};
}

#endif // include guard