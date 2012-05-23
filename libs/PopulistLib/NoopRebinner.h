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
#ifndef _CODE_LIBS_POPULISTLIB_NOOPREBINNER_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_NOOPREBINNER_INCLUDE_GUARD

#include "AbstractRebinner.h"

namespace PopulistLib {

	//! A rebinner that does nothing.
	//!
	//! This can be useful for developing and debugging purposes.
	class NoopRebinner : public AbstractRebinner 
	{
	public: 

		NoopRebinner(){}

		//! virtual destructor
		virtual ~NoopRebinner(){}

		virtual bool Configure
		(	
			valarray<double>&,	//!< reference to density array
			Index,				//!< reversal bin,
			Index,				//!< reset bin
			Number,				// number of  bins before rebinning
			Number				// number of  bins after rebinning
		)
		{
			return true;
		}

		//! rebinning operator, which does nothing
		virtual bool Rebin(AbstractZeroLeakEquations* = 0)
		{
			return true;
		}

		//! mandatory cloning operation
		virtual NoopRebinner* Clone() const
		{
			return new NoopRebinner(*this);
		}

		//! class name
		virtual string Name() const {return string("NoopRebinner");};
	};
}

#endif // include guard
