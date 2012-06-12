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

#ifndef MPILIB_ABSTRACTREPORTHANDLER_HPP_
#define MPILIB_ABSTRACTREPORTHANDLER_HPP_

#include <string>
#include <MPILib/include/BasicTypes.hpp>
#include <DynamicLib/Report.h>

namespace MPILib {

//! Base class for all ReportHandlers
//!
//! ReportHandlers are responsible for dispatching the Reports from each node and collating them
//! in a simulation results file. There are not many prescriptions for how this should be done and
//! it's very simple to derive one's own. AsciiReportHandler records the simulation results in an XML format.
//! RootReportHandler directly stores graphs of simulations. AsciiReportHandler and RootReportHandler come with MIIND.
class AbstractReportHandler {
public:

	//! Takes the file name as argument
	AbstractReportHandler(const std::string& stream_name) :
			_stream_name(stream_name) {
	}

	//! Manadatory virtual destructor
	virtual ~AbstractReportHandler(){};

	virtual bool WriteReport(const DynamicLib::Report&) = 0;

	//! Mandatory cloning operation.
	virtual AbstractReportHandler* Clone() const = 0;

	//! During Configuration a DynamicNode will associate itself with the handler.
	virtual void InitializeHandler(const NodeId&) = 0;

	//! A DynamicNode will request to be dissociated from the handler at the end of simulation.
	virtual void DetachHandler(const NodeId&) = 0;

	std::string MediumName() const {
		return _stream_name;
	}

	//! Default is a NOOP. In a RootReportHandler this function is used.
	virtual void AddNodeToCanvas(NodeId) const {
	}

protected:

private:

	const std::string _stream_name;

};

}// end of MPILib

#endif // MPILIB_ABSTRACTREPORTHANDLER_HPP_ include guard
