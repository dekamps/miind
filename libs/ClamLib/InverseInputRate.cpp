// Copyright (c) 2005 - 2009 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include "ClamLibException.h"
#include "InverseInputRate.h"

using namespace DynamicLib;

Rate ClamLib::InverseInputRate
(
	Rate rate, 
	const WilsonCowanParameter& parameter
) 
{
	// Purpose:
	// to get the input node to a Rate rate a certain input is required,
	// namely the inverse of the squashing function
	// Author: Marc de Kamps
	// Date: 04-05-2007

	// if rate is is outside the range of the squashing function, there is no point in continuing
	Rate f_max = parameter._rate_maximum; 
	if (rate > f_max || rate < 0)
		throw ClamLibException("Input rate out of range");

	// the extremes of the squashing must be covered 
	Rate rate_return;
	if (rate == f_max)
	{
		rate_return = std::numeric_limits<Rate>::max();
		return rate_return;
	}
	if (rate == 0)
	{
		rate_return = -std::numeric_limits<Rate>::max();
		return rate_return;
	}

	Rate beta = parameter._f_noise;

	rate_return = -1/beta*log((2*f_max - rate - 1)/(rate+ 1));
	return rate_return;
}
