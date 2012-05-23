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
#ifndef _CODE_LIBS_POPULISTLIB_AEIFPARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_AEIFPARAMETER_INCLUDE_GUARD

#include "BasicDefinitions.h"

namespace PopulistLib {

	//! Parameters for the adaptive exponential integrate and fire model
	//! as described by Brette & Gerstner (2005)

	struct AEIFParameter {
		Capacity	_C_m;	//!< membrane capacity
		Conductance _g_l;	//!< leak conductance
		Potential	_E_l;	//!< leak potential
		Potential	_V_t;	//!< threshold potential
		Potential   _V_r;	//!< reset potentialion 
		Potential   _D_t;	//!< slope factor
		Time		_t_w;	//!< adaptation time constant
		Conductance _a;		//!< subthreshold adaptation
		Current     _b;		//!< spike-triggered adaptation

		AEIFParameter
		(
			Capacity	C_m = BG_C_M,
			Conductance g_l = BG_G_L,
			Potential	E_l = BG_E_L,
			Potential	V_t = BG_V_T,
			Potential   V_r = BG_V_R,
			Potential   D_t = BG_D_T,
			Time        t_w = BG_T_W,
			Conductance a   = BG_A,
			Current     b   = BG_B
		);

		Number StateDimension() const;
	};
}

#endif // include guard