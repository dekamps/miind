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
#ifndef _CODE_LIBS_NUMTOOLS_EXPONENTIALDISTRIBUTION_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLS_EXPONENTIALDISTRIBUTION_INCLUDE_GUARD

#include "UniformDistribution.h"

// Date:   12-04-1999
// Author: Marc de Kamps
// Short Description: Generates an exponential disribution

namespace NumtoolsLib
{

	//! Generates Exponential distributed values

	class ExponentialDistribution : public Distribution 
	{
	public:
		
		//! Default constructor, needs to refer to a RandomGenerator
		ExponentialDistribution	(RandomGenerator&);

		//! virtual destructor
		virtual	~ExponentialDistribution ();

		//! next sample value

		virtual	double NextSampleValue();

	private:

		// copying of distributions is potentially dangerous (seed mixing, reproduction problems)
		// and seems unnecessary

		ExponentialDistribution	(const ExponentialDistribution&);
		ExponentialDistribution& operator=( const ExponentialDistribution& );			

		UniformDistribution  _uniform_distribution;  //uniform distribution

	}; // end of Numtools

	inline ExponentialDistribution::ExponentialDistribution(RandomGenerator& generator):
	Distribution(generator),
	_uniform_distribution(generator)
	{
	}

	inline ExponentialDistribution::~ExponentialDistribution()
	{
	}

	inline double ExponentialDistribution::NextSampleValue()
	{
		double f_uniform;

		do
			f_uniform = _uniform_distribution.NextSampleValue();

		while ( f_uniform == 0 );

		return -log(f_uniform);
	}

} // end of Numtools

#endif // include guard
