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

#ifndef MPILIB_UTILITIES_LOG_HPP_
#define MPILIB_UTILITIES_LOG_HPP_

#include <iostream>
#include <memory>
#include <time.h>
#include <string>
#include <sstream>

namespace MPILib {
namespace utilities {

/*! \page logging The Log utilities provided by miind
 * This page contains the following sections:
 * <ol>
 * <li>\ref logging_introduction</li>
 * <li>\ref advanced_use</li>
 * <li>\ref details_macro</li>
 * <li>\ref provided_error_levels</li>
 * </ol>
 * \section logging_introduction Introduction
 * To log a message in miind use the following macro:
 * @code{.cpp}
 * LOG(utilities::logWARNING)<<"blub: "<<42;
 * @endcode
 * This would then log a message of the level logWARNING if the current reporting level
 * is higher that logWARNING. Otherwise the logging would be ignored. As this check is
 * done at compile time you pay only for log messages if they are actually printed.
 *
 * \section advanced_use Advanced use of logging
 * The default logging level is \c utilities::logINFO then everything is printed to the log.
 * To change the reporting level of the log class the following code is needed:
 *
 * @code{.cpp}
 * 	utilities::Log::setReportingLevel(utilities::logWARNING);
 * @endcode
 *
 * This code would set the reporting level to \c logWARNING
 *
 * In the default version log messages are printed to std::cerr. To print the log messages
 * into a log file the following code is needed:
 *
 * @code{.cpp}
 * 	std::shared_ptr<std::ostream> pStream( new std::ofstream("MYLOGFILENAME"));
 *	if (!pStream){
 *         throw utilities::Exception("cannot open log file.");
 *	}
 *	utilities::Log::setStream(pStream);
 * @endcode
 *
 * This code would redirect the log messages to the file with the name MYLOGFILENAME.
 *
 * \section details_macro The Log Macro
 *
 * The macro allows easier generation of log messages.
 * It also improves the efficiency significantly as the checks are conducted at compile time.
 * @attention do not pass functions to this macro. This is due to the problematic macro expansion.
 * For example this:
 *
 * @code{.cpp}
 * LOG(logERROR)<<getNumber();
 * @endcode
 *
 * is forbidden. Please use then instead a temporary variable:
 *
 * @code{.cpp}
 * int number = getNumber()
 * LOG(logERROR)<<number;
 * @endcode
 *
 * or alternatively write it the following way:
 *
 * @code{.cpp}
 * if(level > utilities::Log::getReportingLevel() || !utilities::Log::getStream()){
 *     ;
 * }else{
 *     utilities::Log().writeReport(level)<<getNumber;
 * }
 * @endcode
 * However try to use the macro with temporary variables.
 *
 * \section provided_error_levels Provided Error levels
 *
 * <dl>
 * <dt>logERROR</dt>
 * <dd>Only use this for real error messages, as these are always logged.</dd>
 * <dt>logWARNING</dt>
 * <dd>use this for warnings messages.</dd>
 * <dt>logINFO</dt>
 * <dd>Use this for information messages</dd>
 * <dt>logDEBUG</dt>
 * <dd>Use this for very important debug messages</dd>
 * <dt>logDEBUG1</dt>
 * <dd>Use this for important debug messages</dd>
 * <dt>logDEBUG2</dt>
 * <dd>Use this for not so important debug messages</dd>
 * <dt>logDEBUG3</dt>
 * <dd>Use this for not important debug messages</dd>
 * <dt>logDEBUG4</dt>
 * <dd>Use this for for every debug detail messages</dd>
 *
 */

/**
 * The log levels for more details see \ref provided_error_levels
 */
enum LogLevel {
	logERROR,  //!< logERROR
	logWARNING,//!< logWARNING
	logINFO,   //!< logINFO
	logDEBUG,  //!< logDEBUG
	logDEBUG1, //!< logDEBUG1
	logDEBUG2, //!< logDEBUG2
	logDEBUG3, //!< logDEBUG3
	logDEBUG4  //!< logDEBUG4
};

/**
 * @brief class for logging reports. The usage of this log class is described on page \ref logging
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
	static std::shared_ptr<std::ostream> getStream();

	/**
	 * Setter for the stream
	 * @param pStream The Stream the log class should print to.
	 */
	static void setStream(std::shared_ptr<std::ostream> pStream);

	/**
	 * getter for the current report level
	 * @return the current report level
	 */
	static LogLevel getReportingLevel();

	/**
	 * setter for the report level
	 * @param level The new report level
	 */
	static void setReportingLevel(LogLevel level);

private:


	/**
	 * writes the output to the stream
	 * @param msg The message written to the stream
	 */
	static void writeOutput(const std::string& msg);

	/**
	 * The current reporting level of the Log, all messages with a level below this level
	 * are printed the rest is ignored
	 */
	static LogLevel _reportingLevel;

	/**
	 * pointer to the stream
	 */
	static std::shared_ptr<std::ostream> _pStream;
	/**
	 * The buffer for the log message
	 */
	std::ostringstream _buffer;
};

} /* namespace utilities */

/**
 * macro to allow easier generation of log messages.
 * this will improve the efficiency significantly as the checks are conducted at compile time.
 * see also \ref details_macro for more details
 *
 *
 */
#define LOG(level) \
if (level > utilities::Log::getReportingLevel() || !utilities::Log::getStream()) ; \
else utilities::Log().writeReport(level)

} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_LOG_HPP_ */
