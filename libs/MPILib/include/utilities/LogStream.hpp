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
#ifndef MPILIB_UTILITIES_LOGSTREAM_HPP_
#define MPILIB_UTILITIES_LOGSTREAM_HPP_

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <MPILib/include/utilities/Timer.hpp>


//!Util
namespace MPILib
{
namespace utilities{
	//! LogStream
	class LogStream
	{
	public:

		//! Create a LogStream with a closed stream (kind of /dev/null)
		LogStream();            

		//! Associate a LogStream with an ostream; the ostream resource should be created on the heap
		LogStream(boost::shared_ptr<std::ostream>);

		//! Provide sensible behaviour for copying an object containing a LogStream: 
		//! dissociate from the stream of the originator object
		LogStream(const LogStream&);

		//! virtual destructor
		virtual ~LogStream();

		boost::shared_ptr<std::ostream> Stream() const;

		virtual void Record(const std::string&);

		//! Open a stream that was previously closed
		//! if a stream was already open, OpenStream will return false and nothing will be done
		bool OpenStream(boost::shared_ptr<std::ostream>);

		//! Is an open stream associated with this LogStream
		bool IsOpen() const;

		void flush();

		//! In exceptional cases it is sometimes better to write a log file and then make sure it's closed
		//! Calling close disengages the Logstream completely from its associated ostream
		void close();

		friend  LogStream& operator<<( LogStream&, const char* );
		friend  LogStream& operator<<( LogStream&, const std::string& );
		friend  LogStream& operator<<( LogStream&, int );
		friend  LogStream& operator<<( LogStream&, double );
	
	private:
		


		boost::shared_ptr<std::ostream>		_p_stream_log;     // pointer to the log stream

		bool							_b_time_available; // system time available ?

		Timer							_timer;            // timer, records time between calls of Record


	}; // end of LogStream

	LogStream& operator<<( LogStream&, const char* );
	LogStream& operator<<( LogStream&, const std::string& );
	LogStream& operator<<( LogStream&, int );
	LogStream& operator<<( LogStream&, double );

} //end utilities
} // end of MPILib


#endif //MPILIB_UTILITIES_LOGSTREAM_HPP_ include guard
