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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NUMTOOLSLIB_RANDOMGENERATOR_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_RANDOMGENERATOR_INCLUDE_GUARD

#include <vector>
#include <iostream>
#include <gsl/gsl_rng.h>
#include "ChangeSeedException.h"

using std::vector;

namespace NumtoolsLib
{

	//! Global default seed for RandomGenerator

	const long GLOBAL_SEED = 9875987L;


	//! This class contains a specific uniform random generator that can be used to generate other distributions
	//! GaussianDistribution and UniformDistribution, for example, use this random generator, which gsl_rng_mt19937
	//! of the Gnu Scietific Library. This class can be used to modify the seed and to keep track of the number
	//! of draws made by the generator. Distributions use one copy of this class.

	class RandomGenerator 
	{

	public:

		RandomGenerator(long lseed = GLOBAL_SEED);
		~RandomGenerator();

		double	NextSampleValue();

	private:
	 

		RandomGenerator(const RandomGenerator&);
		RandomGenerator& operator=(const RandomGenerator&);

		double	Ran2(long*);
	
		// new, uniformly distributed value

		void    AddOne         (){ std::cout << "zopa" << std::endl; ++_number_of_draws; }


		long			_initial_seed;    // initial seed value
		unsigned long	_number_of_draws; // number of calls to NextSampleValue

		gsl_rng*		_p_generator;

	
	}; // end of RandomGenerator


	//! Global Random Generator
	extern RandomGenerator GLOBAL_RANDOM_GENERATOR;

	inline double RandomGenerator::NextSampleValue()
	{
		AddOne();
		return gsl_rng_uniform(_p_generator);
	}




} // end of Numtools


#endif // include guard
