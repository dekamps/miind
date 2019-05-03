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

#ifndef MPILIB_DELAYEDCONNECTIONQUEUE_HPP_
#define MPILIB_DELAYEDCONNECTIONQUEUE_HPP_

#include <deque>
#include <MPILib/include/TypeDefinitions.hpp>
#include <cmath> // changed to cmath (MdK: 8/4/2019)

namespace MPILib {

/**
* @brief This is a carbon copy of the DelayAlgorithm class.
*/
class DelayedConnectionQueue {
public:

 /**
  * Create algorithm with a delay time
  * @param t_delay The delay time
  */
 DelayedConnectionQueue(Time timestep = 0.001,Time delay = 0):
   	_t_delay(delay),
    _queue(1 + static_cast<int>(std::floor(delay/timestep))),
    _t_delay_proprtion(std::abs(std::fmod(delay,timestep) - timestep) < 0.0000000001 ? 0.0 : std::fmod(delay,timestep)/timestep){
    }

 void updateQueue(ActivityType inRate);

 ActivityType getCurrentRate() const;

 Time getDelayTime() const { return _t_delay; }

private:

 Time _t_delay;
 Time _t_delay_proprtion;
 ActivityType _rate_current;

 std::deque<ActivityType> _queue;

};

} /* end namespace MPILib */

#endif
