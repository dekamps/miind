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
#ifndef _CODE_LIBS_CLAMLIB_DYNAMICSUBNETWORKITERATOR_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_DYNAMICSUBNETWORKITERATOR_INCLUDE_GUARD

#include "DynamicSubNetwork.h"

namespace ClamLib {

	class SimulationResultIterator {
	public:

		SimulationResultIterator(vector<DynamicSubNetwork>::iterator iter):_iter(iter){}

		DynamicSubNetwork& operator*(){return *_iter; }

		DynamicSubNetwork* operator->(){ return &(*_iter); }

			//! Define the customary iterator increment (prefix)
		SimulationResultIterator& operator++(){ ++_iter; return *this;}


		//! Define the customary iterator increment (postfix)
		SimulationResultIterator  operator++(int)	
		{
			// postfix
			SimulationResultIterator iter = *this;
			++*this;
			return iter;	
		}
	friend bool operator != 
		(
			const SimulationResultIterator&,
			const SimulationResultIterator&
		);

	private:

		vector<DynamicSubNetwork>::iterator _iter;
	};

	inline bool operator!=( const SimulationResultIterator& it1, const SimulationResultIterator& it2)
	{
		return (it1._iter != it2._iter);
	}
}

#endif // include guard
