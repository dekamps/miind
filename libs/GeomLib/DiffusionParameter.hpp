// Copyright (c) 2005 - 2014 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification,are permitted provided that the following conditions are
// met
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of
//      conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived
//      from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _CODE_LIBS_GEOMLIB_DIFFUSIONPARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_DIFFUSIONPARAMETER_INCLUDE_GUARD



namespace GeomLib {

	//! When to switch to a two Poisson input approximation, and what input jump to use then.

	//! It is often useful to emulate Gaussian white noise with a Poisson input. For
	//! low variability white noise, a single Poisson input is sufficient. When the variability
	//! is high, two inputs must be used. This parameter determines when the switch is made (diffusion_limit)
	//! and what efficacy is being used for the two Poisson inputs (diffusion jump). The
	//! the values are typically interpreted as percentages of a scale, which is set by the neural model.

	struct DiffusionParameter {

		double _diffusion_jump;
		double _diffusion_limit;

		DiffusionParameter
		(
			double diffusion_jump  = 0.03,
			double diffusion_limit = 0.03
		):
		_diffusion_jump(diffusion_jump),
		_diffusion_limit(diffusion_limit)
		{
		}

	};
} // namespace

#endif // include guard
