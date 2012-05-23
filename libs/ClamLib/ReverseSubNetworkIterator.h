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
#ifndef _CODE_LIBS_CLAMLIMLIB_REVERSESUBNETWORKITERATOR_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIMLIB_REVERSESUBNETWORKITERATOR_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4267)
#pragma warning(disable: 4996)
#endif 

#include <vector>
#include "../StructnetLib/StructnetLib.h"
#include "CircuitInfo.h"
#include "ClamLibException.h"
#include "RootLayeredNetDescription.h"
#include "ToLayeredDescriptionVector.h"

using std::vector;
using StructnetLib::NodeIdPosition;
using StructnetLib::PhysicalPosition;
using StructnetLib::RZOrder;

namespace ClamLib {

	class ReverseSubNetworkIterator {
	public:
		ReverseSubNetworkIterator
		(
			vector<CircuitInfo>::const_iterator iter,
			const vector<CircuitInfo>&			vec_circuit,
			const RootLayeredNetDescription&	desc
		);

		ReverseSubNetworkIterator(const ReverseSubNetworkIterator&);

		const CircuitInfo* operator->(){if (_iter < _vec_circuit.end()) return &(*_iter);  else throw ClamLibException(EXCEP);}

		const CircuitInfo& operator*(){ return *_iter; }

		bool Position(PhysicalPosition* pos) const;

		//! Define the customary iterator increment (prefix)
		ReverseSubNetworkIterator& operator++()
		{
			// first update the order
			++_rz_order;
			// then use it to point to the right vector element
			if ( _rz_order.Id()._id_value == -1)
				_iter = _vec_circuit.end();
			else {
				_iter = _vec_circuit.begin() +  _rz_order.Id()._id_value;
				// and check whether it actaully does
				assert( _iter->IdOriginal()._id_value == _rz_order.Id()._id_value);
			}
	
			return *this;
		}

		//! Define the customary iterator increment (postfix)
		ReverseSubNetworkIterator  operator++(int)	
		{
			// postfix
			ReverseSubNetworkIterator iter = *this;
			++*this;
			return iter;	
		}

		const RootLayeredNetDescription& LayeredNetDescription() const { return _desc; };

		friend bool operator!=(const ReverseSubNetworkIterator&, const ReverseSubNetworkIterator&);

		vector<CircuitInfo>::const_iterator iter() const { return _iter; }
		vector<CircuitInfo>::const_iterator iterEnd() const { return _iter_end; }

	private:

		vector<CircuitInfo>::const_iterator _iter;
		vector<CircuitInfo>::const_iterator _iter_end;
		const vector<CircuitInfo>&			_vec_circuit;
		NodeIdPosition						_id_pos;
		RZOrder								_rz_order;
		const RootLayeredNetDescription		_desc;

		static const string EXCEP;
	};

	inline bool operator!=(const ReverseSubNetworkIterator& it1, const ReverseSubNetworkIterator& it2)
	{
		return (it1._iter != it2._iter);
	}
}
	

#endif // include guard
