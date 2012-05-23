// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_CLAMLIB_FORWARDSUBNETWORKITERATOR_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_FORWARDSUBNETWORKITERATOR_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4267)
#pragma warning(disable: 4996)
#endif

#include <vector>
#include "CircuitInfo.h"
#include "ToRootLayeredNetDescription.h"
#include "RootLayeredNetDescription.h"
#include "ToLayeredDescriptionVector.h"
#include "../StructnetLib/StructnetLib.h"

using StructnetLib::NodeIdPosition;
using StructnetLib::PhysicalPosition;
using std::string;
using std::vector;


namespace ClamLib {

	//! This iterator will traverse the network from input layer to output layer
	class ForwardSubNetworkIterator {
	public:


		ForwardSubNetworkIterator
		(
			vector<CircuitInfo>::const_iterator iter,
			vector<CircuitInfo>::const_iterator iter_end,
			const RootLayeredNetDescription&	desc
		):
		_iter(iter),
		_iter_end(iter_end),
		_id_pos(ToLayeredDescriptionVector(desc)),
		_desc(desc)
		{
		}

		const CircuitInfo* operator->(){ if (_iter < _iter_end) return &(*_iter); else throw ClamLibException(EXCEP);}

		const CircuitInfo& operator*(){ if (_iter < _iter_end) return *_iter; else throw ClamLibException(EXCEP); }

		bool Position(PhysicalPosition* pos) const;

		//! Define the customary iterator increment (prefix)
		ForwardSubNetworkIterator& operator++()
		{
			++_iter;
			return *this;
		}

		//! Define the customary iterator increment (postfix)
		ForwardSubNetworkIterator  operator++(int)	
		{
			// postfix
			ForwardSubNetworkIterator iter = *this;
			++*this;
			return iter;	
		}

		const RootLayeredNetDescription& LayeredNetDescription() const { return _desc; };

		friend bool operator!=(const ForwardSubNetworkIterator&, const ForwardSubNetworkIterator&);

		vector<CircuitInfo>::const_iterator iter() const { return _iter; }
		vector<CircuitInfo>::const_iterator iterEnd() const { return _iter_end; }
	private:

		vector<CircuitInfo>::const_iterator _iter;
		vector<CircuitInfo>::const_iterator _iter_end;
		NodeIdPosition						_id_pos;

		const RootLayeredNetDescription		_desc;


		static const string EXCEP;
	};

	inline bool operator!=(const ForwardSubNetworkIterator& it1, const ForwardSubNetworkIterator& it2)
	{
		return (it1._iter != it2._iter);
	}
}

#endif // include guard
