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
#include <MPILib/include/utilities/LogStream.hpp>
#include <MPILib/include/utilities/TimeException.hpp>

namespace MPILib {
namespace utilities {

LogStream::LogStream() {
}

LogStream::LogStream(std::shared_ptr<std::ostream> p_stream_log) :
		_pStreamLog(p_stream_log) {

	try {
		// just check if there is timing
		float time_first = _timer.secondsSinceLastCall();
		time_first+=1.0;

	} catch (TimeException &e) {
		_isTimeAvailable = false;
		*_pStreamLog << "No time available error: "<<e.what() << std::endl;

	}
}

LogStream::~LogStream() {
	if (_pStreamLog)
		*_pStreamLog << "Total time: " << _timer.secondsSinceFirstCall()
				<< std::endl;
}

void LogStream::record(const std::string& string_message) {

	if (_pStreamLog) {
		if (_isTimeAvailable)
			*_pStreamLog << _timer.secondsSinceLastCall() << "\t"
					<< string_message << std::endl;
		else
			*_pStreamLog << string_message << std::endl;
	}
}

std::shared_ptr<std::ostream> LogStream::getStream() const {
	return _pStreamLog;
}

void LogStream::flush() {
	if (_pStreamLog)
		_pStreamLog->flush();
}

void LogStream::close() {
	// flush the stream's buffer 
	if (_pStreamLog)
		_pStreamLog->flush();
}

bool LogStream::openStream(std::shared_ptr<std::ostream> p_stream) {
	if (_pStreamLog)
		return false;
	else {
		_pStreamLog = p_stream;
		return true;
	}
}

LogStream& operator<<(LogStream& stream, const char* p) {
	if (stream._pStreamLog)
		*stream._pStreamLog << p;
	return stream;
}

LogStream& operator<<(LogStream& stream, const std::string& message) {
	if (stream._pStreamLog)
		*stream._pStreamLog << message;

	// else
	// no op (/dev/null)
	return stream;
}

LogStream& operator<<(LogStream& stream, double f) {
	if (stream._pStreamLog)
		*stream._pStreamLog << f;
	// else
	// no op (/dev/null)

	return stream;
}

LogStream& operator<<(LogStream& stream, int i) {
	if (stream._pStreamLog)
		*stream._pStreamLog << i;
	// else
	// no op (/dev/null)

	return stream;
}
}
}
