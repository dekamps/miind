// Copyright (c) 2005 - 2010 Marc de Kamps
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

#include <MPILib/include/utilities/LogStream.hpp>
#include <MPILib/include/utilities/TimeException.hpp>

namespace MPILib {
namespace utilities {

LogStream::LogStream() {
}

LogStream::LogStream(std::shared_ptr<std::ostream> p_stream_log) :
		_p_stream_log(p_stream_log), _b_time_available(true) {

	try {
		// just check if there is timing
		float time_first = _timer.SecondsSinceLastCall();

	} catch (TimeException &e) {
		_b_time_available = false;
		*_p_stream_log << "No time available error: "<<e.what() << std::endl;

	}
}

LogStream::~LogStream() {
	if (_p_stream_log)
		*_p_stream_log << "Total time: " << _timer.SecondsSinceFirstCall()
				<< std::endl;
}

void LogStream::Record(const std::string& string_message) {

	if (_p_stream_log) {
		if (_b_time_available)
			*_p_stream_log << _timer.SecondsSinceLastCall() << "\t"
					<< string_message << std::endl;
		else
			*_p_stream_log << string_message << std::endl;
	}
}

std::shared_ptr<std::ostream> LogStream::Stream() const {
	return _p_stream_log;
}

void LogStream::flush() {
	if (_p_stream_log)
		_p_stream_log->flush();
}

void LogStream::close() {
	// flush the stream's buffer 
	if (_p_stream_log)
		_p_stream_log->flush();
}

bool LogStream::OpenStream(std::shared_ptr<std::ostream> p_stream) {
	if (_p_stream_log)
		return false;
	else {
		_p_stream_log = p_stream;
		return true;
	}
}

LogStream& operator<<(LogStream& stream, const char* p) {
	if (stream._p_stream_log)
		*stream._p_stream_log << p;
	return stream;
}

LogStream& operator<<(LogStream& stream, const std::string& message) {
	if (stream._p_stream_log)
		*stream._p_stream_log << message;

	// else
	// no op (/dev/null)
	return stream;
}

LogStream& operator<<(LogStream& stream, double f) {
	if (stream._p_stream_log)
		*stream._p_stream_log << f;
	// else
	// no op (/dev/null)

	return stream;
}

LogStream& operator<<(LogStream& stream, int i) {
	if (stream._p_stream_log)
		*stream._p_stream_log << i;
	// else
	// no op (/dev/null)

	return stream;
}
}
}
