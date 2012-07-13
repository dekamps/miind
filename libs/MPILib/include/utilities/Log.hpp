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
#include <sstream>

namespace MPILib {
namespace utilities {

/**
 * The log levels
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
 * @brief class for logging reports.
 *
 * The class Log allows easy logging of messages. It allows to log messages of different
 * levels. It logs only messages where the level is lower than the  current ReportingLevel of this
 * class. The default level is logDEBUG4 which logs everything. The check is done at compile time
 * therefore the overhead is very low.
 *
 * To set the reporting level the following code is needed:
 * @code
 * 	utilities::Log::setReportingLevel(utilities::logWARNING);
 * @endcode
 *
 * In the default version log messages are printed with std::cerr. To print the log messages
 * into a log file the following code is needed:
 * @code
 * 	std::ostream* pStream = new std::ofstream("MYLOGFILENAME");
 *	if (!pStream)
 *		throw utilities::Exception("cannot open log file.");
 *	utilities::Log::getStream()=pStream;
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

	/**
	 * getter for the current report level
	 * @return
	 */
	static LogLevel getReportingLevel();

	/**
	 * setter for the report level
	 * @param level The new report level
	 */
	static void setReportingLevel(LogLevel level);

private:
	/**
	 * The current reporting level of the Log, all messages with a level below this level
	 * are printed the rest is ignored
	 */
	static LogLevel _reportingLevel;

	/**
	 * The buffer for the log message
	 */
	std::ostringstream _buffer;
};

} /* namespace utilities */

/**
 * macro to allow easier generation of log messages.
 * this will improve the efficiency significantly as the checks are conducted at compile time.
 */
#define LOG(level) \
if (level > utilities::Log::getReportingLevel() || !utilities::Log::getStream()) ; \
else utilities::Log().writeReport(level)

} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_LOG_HPP_ */
