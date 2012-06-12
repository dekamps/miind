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

#include <MPILib/include/utilities/Timer.hpp>
#include <MPILib/include/utilities/TimeException.hpp>

namespace MPILib{
namespace utilities{



static const size_t SECONDS_PER_HOUR		= 3600;

Timer::Timer():
_time_at_first_call( time(&_time_since_last_call) )
{

	if ( _time_at_first_call == -1 )
		throw TimeException("Error during call of Timer Constructor");
}



float Timer::SecondsSinceLastCall()
{
	time_t time_dummy, time_now = time(&time_dummy);
	
	float ret = static_cast<float>(difftime(time_now, _time_since_last_call));
	_time_since_last_call = time_now;

	return ret;
}


float Timer::HoursSinceLastCall()
{
	float f_ret = SecondsSinceLastCall();

	return f_ret/SECONDS_PER_HOUR;
}

float Timer::SecondsSinceFirstCall()
{
	time_t time_dummy, time_now = time(&time_dummy);
	return static_cast<float>(difftime(time_now, _time_at_first_call));
}


}
}
