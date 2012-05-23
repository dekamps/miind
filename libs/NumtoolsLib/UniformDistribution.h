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
#ifndef _CODE_LIBS_NUMTOOLSLIB_UNIFORMDISTRIBUTION_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_UNIFORMDISTRIBUTION_INCLUDE_GUARD

// Date:   12-04-1999
// Author: Marc de Kamps
// Short Description:  Produces random value, uniformly distributed in [0,1)
 
#include "Distribution.h"


namespace NumtoolsLib
{
	//! UniformDistribution
	//! NextSampleValue generates random values that are uniformly distributed
	//! in the interval [0,1]. 

	class UniformDistribution : public Distribution 
	{
	public:

		UniformDistribution(RandomGenerator&);

		//! Generate next sample value

		virtual	double	NextSampleValue();

	private:
			
		// copying of distributions is potentially dangerous (seed mixing, reproduction problems)
		// and seems unnecessary

		UniformDistribution	(const UniformDistribution&);
		UniformDistribution& operator=(const UniformDistribution&);			

	}; // end of UniformDistribution

	inline double UniformDistribution::NextSampleValue()
	{
		return Distribution::NextSampleValue();
	}


} // end of Numtools

#endif // include gaurd

