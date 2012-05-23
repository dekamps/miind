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
#ifndef _CODE_LIBS_POPULISTLIB_SINGLEPEAKREBINNER_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_SINGLEPEAKREBINNER_INCLUDE_GUARD

#include "../UtilLib/UtilLib.h"
#include "AbstractRebinner.h"
#include "AbstractZeroLeakEquations.h"

using UtilLib::Index;
using UtilLib::Number;

namespace PopulistLib {

	//! SinglePeakRebinner is a test class, used for debugging but not as a serious rebinner
	//!
	//! SinglePeak functions under the assumption that the density profile is just concentrated
	//! in a single peak. This makes it possible to study the effect of a rebinning operation:
	//! the peak should be weher it's supposed to be
	class SinglePeakRebinner : public AbstractRebinner {

	public:

		SinglePeakRebinner();

		virtual ~SinglePeakRebinner();

		virtual bool Configure
			(
				valarray<double>&,
				Index,
				Index,
				Number,
				Number
			);

		//! Rebin operation. SinglePeakRebinner does not take refractive probability into account
		virtual bool Rebin(AbstractZeroLeakEquations*);

		//! Type transfer
		virtual SinglePeakRebinner* Clone() const;

		//! Name of the rebinning class
		virtual string Name() const {return string("SinglePeakRebinner");}

	private:

		valarray<double>* _p_array;        
		Index             _index_reversal_bin; 
		Index			  _index_reset_bin;
		Number            _number_original_bins;
		Number            _number_new_bins;    
	};
}

#endif // include guard
