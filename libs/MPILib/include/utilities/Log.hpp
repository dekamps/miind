/*
 * Log.hpp
 *
 *  Created on: 12.07.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_LOG_HPP_
#define MPILIB_UTILITIES_LOG_HPP_

#include <iostream>
#include <memory>
#include <time.h>
#include <string>
#include <MPILib/include/utilities/Exception.hpp>

namespace MPILib {
namespace utilities {

/**
 * The debug levels
 */
enum LogLevel {
	logERROR,  //!< logERROR
	logWARNING,  //!< logWARNING
	logINFO,   //!< logINFO
	logDEBUG,  //!< logDEBUG
	logDEBUG1, //!< logDEBUG1
	logDEBUG2, //!< logDEBUG2
	logDEBUG3, //!< logDEBUG3
	logDEBUG4  //!< logDEBUG4
};

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

/**
 * implements a output policy for the Log class
 */
class Output2FILE {
public:
	/**
	 * The Stream it writes to the standard is std::cerr
	 * @return The Stream it writes to
	 */
	static std::ostream*& getStream();

	/**
	 * writes the output to the stream
	 * @param msg The message written to the stream
	 */
	static void writeOutput(const std::string& msg);
};

inline std::ostream*& Output2FILE::getStream() {
	static std::ostream* pStream = &std::cerr;
	return pStream;
}
inline void Output2FILE::writeOutput(const std::string& msg) {
	std::ostream* pStream = getStream();
	if (!pStream)
		throw utilities::Exception(
				"the stream is not available. There must have an error occurred.");
	(*pStream) << msg;
	pStream->flush();
}

/**
 * @brief class for logging reports.
 *
 * The class Log allows easy logging of messages. It allows to log messages of different
 * levels. It logs only messages where the level is lower than the  current ReportingLevel of this
 * class. The default level is logDEBUG4 which logs everything. The check is done at compile time
 * therefore the overhead is very low.
 *
 * To set the reporting level the following code is needed:
 * @code
 * 	FILELog::ReportingLevel = utilities::logWARNING;
 * @endcode
 *
 * In the default version log messages are printed with std::cerr. To print the log messages
 * into a log file the following code is needed:
 * @code
 * 	std::ostream* pStream = new std::ofstream("MYLOGFILENAME");
 *	if (!pStream)
 *		throw utilities::Exception("cannot open log file.");
 *	utilities::Output2FILE::getStream()=pStream;
 * @endcode
 *
 * To actually log a message the following macro can be used:
 * @code
 * LOG(utilities::logWARNING)<<"blub: "<<42;
 * @endcode
 * This code would log a message of the level logWARNING if the current reporting level
 * is higher that logWARNING.
 *
 *
 *
 */
template<typename OutputPolicy>
class Log {
public:
	/**
	 * default constructor
	 */
	Log()=default;
	/**
	 * copy constructor deleted
	 */
	Log(const Log&)=delete;
	/**
	 * copy operator deleted
	 */
	Log& operator =(const Log&)=delete;
	/**
	 * destructor which writes the message to the stream
	 */
	virtual ~Log();

	/**
	 * takes the log message and stores it in the buffer
	 * @param level The level of the log message
	 * @return  a ostringstream
	 */
	std::ostringstream& writeReport(LogLevel level = logINFO);
public:
	/**
	 * The current reporting level of the Log, all messages with a level below this level
	 * are printed the rest is ignored
	 */
	static LogLevel ReportingLevel;

	/**
	 * The buffer for the log message
	 */
	std::ostringstream _buffer;
};

/**
 * The default log level is set to the finest possible, everything is logged.
 */
template<typename OutputPolicy>
LogLevel Log<OutputPolicy>::ReportingLevel = logDEBUG4;

template<typename OutputPolicy>
std::ostringstream& Log<OutputPolicy>::writeReport(LogLevel level) {
	//generate time in the format Date HH::MM::SS
	time_t rawtime;
	time(&rawtime);
	struct tm foo;
	struct tm *mytm;
	mytm = localtime_r (&rawtime, &foo);
	char outstr[200];
	strftime(outstr, sizeof(outstr), "%x% %H:%M:%S", mytm);

	_buffer << "- " << outstr;
	_buffer << " " << logLevelToString(level) << ": ";
	return _buffer;
}

template<typename OutputPolicy>
Log<OutputPolicy>::~Log() {
	_buffer << std::endl;
	OutputPolicy::writeOutput(_buffer.str());
}

} /* namespace utilities */

typedef utilities::Log<utilities::Output2FILE> FILELog;
/**
 * macro to allow easier generation of log messages.
 * this will improve the efficiency significantly as the checks are conducted at compile time.
 */
#define LOG(level) \
if (level > FILELog::ReportingLevel || !utilities::Output2FILE::getStream()) ; \
else FILELog().writeReport(level)

} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_LOG_HPP_ */
