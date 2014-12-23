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

#include "GeomParameter.hpp"

using namespace GeomLib;

GeomParameter::GeomParameter
(
	const AbstractOdeSystem& 			sys,
	Potential 							scale,
	const DiffusionParameter& 			par_diff,
	const CurrentCompensationParameter&	par_cur,
	const string& 						name_master,
	bool								no_master_equation
):
_p_sys_ode			(sys.Clone()),
_scale				(scale),
_par_diff			(par_diff),
_par_cur			(par_cur),
_name_master		(name_master),
_no_master_equation	(no_master_equation)
{
}

GeomParameter::GeomParameter
(
	const GeomParameter& par
):
_p_sys_ode			(par._p_sys_ode->Clone()),
_scale				(par._scale),
_par_diff			(par._par_diff),
_name_master		(par._name_master),
_no_master_equation	(par._no_master_equation)
{
}

