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
#include <MPILib/config.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>
#include <MPILib/include/utilities/Log.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#include <iomanip>

namespace MPILib {
namespace utilities {

namespace {
/**
 * helper function to convert debug levels to stings
 * @param level the debug level
 * @return the string of the debug level
 */
std::string logLevelToString(LogLevel& level) {
	switch (level) {
	case logERROR:
		return std::string("Error");
		break;
	case logWARNING:
		return std::string("Warning");
		break;
	case logINFO:
		return std::string("Info");
		break;
	case logDEBUG:
		return std::string("Debug");
		break;
	case logDEBUG1:
		return std::string("Debug1");
		break;
	case logDEBUG2:
		return std::string("Debug2");
		break;
	case logDEBUG3:
		return std::string("Debug3");
		break;
	case logDEBUG4:
		return std::string("Debug4");
		break;
	default:
		break;
	}
	return std::string("");
}

struct null_deleter {
	void operator()(void const *) const {
	}
};
}

/**
 * The default log level is set to the finest possible, everything is logged.
 */
#ifdef DEBUG
	LogLevel Log::_reportingLevel = logDEBUG4;
#else
	LogLevel Log::_reportingLevel = logINFO;
#endif
/**
 * Default the log is printed to std::cerr. To avoid the deletion of std::cerr
 * a null_deleter is provided.
 */
std::shared_ptr<std::ostream> Log::_pStream(&std::cerr, null_deleter());

std::shared_ptr<std::ostream> Log::getStream() {
	return _pStream;
}

void Log::setStream(std::shared_ptr<std::ostream> pStream) {
	_pStream = pStream;
}

void Log::writeOutput(const std::string& msg) {
	std::shared_ptr<std::ostream> pStream = getStream();
	if (!pStream)
		throw utilities::Exception(
				"The stream is not available. There must have an error occurred.");
	(*pStream) << msg;
	pStream->flush();
}

std::ostringstream& Log::writeReport(LogLevel level) {
	//generate time in the format Date HH::MM::SS
	time_t rawtime;
	time(&rawtime);
	struct tm tempTm1;
	struct tm *tempTm2;
	tempTm2 = localtime_r(&rawtime, &tempTm1);
	char outstr[200];
	strftime(outstr, sizeof(outstr), "%x% %H:%M:%S", tempTm2);

	_buffer << "- " << outstr;
	_buffer << " Proc " << MPIProxy().getRank() << " of "
			<< MPIProxy().getSize();
	_buffer << std::setw(10) << logLevelToString(level) << ":\t";
	return _buffer;
}

void Log::setReportingLevel(LogLevel level) {
	LOG(logINFO) << "Report Level changed from "
			<< logLevelToString(_reportingLevel) << " to "
			<< logLevelToString(level);
	_reportingLevel = level;
}

LogLevel Log::getReportingLevel() {
	return _reportingLevel;
}

Log::~Log() {
	_buffer << std::endl;
	Log::writeOutput(_buffer.str());
}

}
}
