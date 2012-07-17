/*
 * Log.cpp
 *
 *  Created on: 13.07.2012
 *      Author: david
 */
#include <MPILib/config.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>
#include <MPILib/include/utilities/Log.hpp>
#include <MPILib/include/utilities/Exception.hpp>

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
LogLevel Log::_reportingLevel = logDEBUG4;
/**
 * Default the log is printed to std::cerr. To avoid the deletion of std::cerr
 * a null_deleter is provided.
 */
std::shared_ptr<std::ostream> Log::_pStream(&std::cerr, null_deleter());

std::shared_ptr<std::ostream> Log::getStream(){
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
	MPIProxy mpiProxy;
	_buffer << " Proc " << mpiProxy.getRank() << " of " << mpiProxy.getSize();
	_buffer << " " << logLevelToString(level) << ":\t";
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
