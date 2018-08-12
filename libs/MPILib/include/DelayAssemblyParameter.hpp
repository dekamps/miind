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

#ifndef MPILIB_ALGORITHMS_DELAYASSEMBLYPARAMETER_HPP_
#define MPILIB_ALGORITHMS_DELAYASSEMBLYPARAMETER_HPP_

#include <limits>
#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {
struct DelayAssemblyParameter {

	DelayAssemblyParameter() {
	}

	/**
	 * constructor for convenience
	 * @param time_membrane membrane time constant in ms
	 * @param rate initial firing rate of the population
	 * @param th_exc excitatory threshold to switch delay activation on
	 * @param  th_inh inhibitory threshold to switch delay off
     * @param  slope amount of firing rate change allowed per time step
	 */

  DelayAssemblyParameter(Time time_membrane,Rate rate, Rate th_exc, Rate th_inh, Rate slope):
    _time_membrane(time_membrane), _rate(rate), _th_exc(th_exc), _th_inh(th_inh), _slope(slope){
	}

	/**
	 * virtual destructor
	 */
	virtual ~DelayAssemblyParameter() {
	}

    /**
	 * membrane time constant; no decay by default
	 */
        Time _time_membrane = std::numeric_limits<Time>::max();

    /**
	 * current firing rate
	 */
        Rate _rate = 0;

    /**
	 * activation level for switching on the assembly
	 */
        Rate _th_exc = 1.0;

    /**
	 * activation level for switching off the assembly
	 */
        Rate _th_inh = 2.0;

    /**
     * amount of change per time step of firing rate
     */
        Rate _slope = 0.1;

     };

} // end of namespaces

#endif // include guard
