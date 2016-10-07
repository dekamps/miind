// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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
#ifdef WANTROOT 

#ifndef MPILIB_REPORT_HANDLER_VALUEVALUEHANDLER_HPP_
#define MPILIB_REPORT_HANDLER_VALUEVALUEHANDLER_HPP_

#include <MPILib/include/report/Report.hpp>

#include <vector>
class TGraph;

namespace MPILib {
namespace report {
namespace handler {

/**
 * ValueHandlerHandler is an auxilliary class for the RootReportHandler which keeps track
 * of quantities that need to be logged in the simulation file and which are registered as such
 * during simulation
 */
class ValueHandlerHandler {
public:

	ValueHandlerHandler()=default;

	/**
	 * Adds a report to the ValueHandlerHandler
	 * @param report A Report
	 */
	void addReport(const Report& report);
	/**
	 * Write the Events to a file
	 */
	void write();

	/**
	 * everything is stored as an event
	 */
	struct Event {
		std::string _str;
		float _time;
		float _value;
	};

	/**
	 * Are the events written to a file
	 * @return true if they were written to file
	 */
	bool isWritten() const;

	/**
	 * resets all Events
	 */
	void reset();

private:

	/**
	 * Stores a event
	 * @param ev a event
	 */
	void distributeEvent(const Event& ev);

	bool _is_written = false;
	std::vector<std::string> _vec_names;
	std::vector<std::vector<float> > _vec_time;
	std::vector<std::vector<float> > _vec_quantity;
};

} // end namespace of handler
} // end namespace of report
} // end namespace of MPILib

#endif // include guard
#endif // don't bother if you don't want ROOT
