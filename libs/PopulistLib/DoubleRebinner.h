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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_POPULISTLIB_DOUBLEREBINNER_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_DOUBLEREBINNER_INCLUDE_GUARD

#include "AbstractRebinner.h"


namespace PopulistLib {

	//! A simple rebinning algorithm for debugging and testing purposes.
	//!
	//! In general one does NOT want to use this algorithm. One has to let the grid expand by exactly a factor of two,
	//! the rebinning algorithm then lumps bins form the expanded grid into one bin of the original grid. In general,
	//! one doesn't want to wait this long and even as a rebinning algorithm it doesn't work fantastically. It is useful
	//! in debugging, however, because it is conceptually the simplest way to rebin.
	class DoubleRebinner : public AbstractRebinner {

	public:
		//! default constructor
		DoubleRebinner();

		//! mandatory virtual destructor
		virtual ~DoubleRebinner();

		virtual bool Configure
			(	
				valarray<double>&,	//!< reference to the density array
				Index,				//!< reversal bin,
				Index,				//!< reset bin
				Number,				//!< number of  bins before rebinning
				Number				//!< number of  bins after rebinning
			);

		//! Do the actual rebinning. Double rebinner does not take into account refractive probability
		virtual bool Rebin(AbstractZeroLeakEquations*);

		//! Mandatory cloning operator
		virtual DoubleRebinner* Clone() const;

		//! The expansion factor is defined by #bins after exp( gamma delta t) =2. Typically this is not 2.#original bins, 
		//! because there are also bins below the reversal potential which are not considered in the expansion.
		double ExpansionFactor() const;

		//! class name
		virtual string Name() const {return string("DoubleRebinner");}


	private:
		void RebinPositive();
		void RebinNegative();

		valarray<double>* _p_array_state;

		int _number_original_growing;
		int _number_expanded_growing;
		int _i_reversal; 
		int _i_odd;
	};
}


#endif // include guard
