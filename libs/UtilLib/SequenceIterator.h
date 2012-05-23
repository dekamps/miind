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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_UTILLIB_SEQUENCEITERATOR_INCLUDE_GUARD
#define _CODE_LIBS_UTILLIB_SEQUENCEITERATOR_INCLUDE_GUARD

#include <cassert>
#include <iterator>
#include "BasicDefinitions.h"

using std::forward_iterator_tag;
using std::iterator;

namespace UtilLib {

	class Sequence;

	class SequenceIterator;
	bool operator==(const SequenceIterator&, const SequenceIterator&);
	bool operator!=(const SequenceIterator&, const SequenceIterator&);

	class SequenceIterator : public iterator<forward_iterator_tag, double>{
	public:

		SequenceIterator():_ind(0),_p_seq(0){}

		SequenceIterator(const Sequence& seq, Index ind):_ind(ind),_p_seq(&seq){}

		SequenceIterator(const SequenceIterator& seq):_ind(seq._ind),_p_seq(seq._p_seq){}

		//! Define the customary iterator increment (prefix)	
		SequenceIterator& operator++()
		{
			++_ind;
			return *this;
		}

		//! Define the customary iterator increment (postfix)	
		SequenceIterator operator++(int)
		{
			SequenceIterator ret = *this;
			++_ind;
			return ret;
		}

		SequenceIterator operator-(int i)
		{
			SequenceIterator ret = *this;
			assert (i <= static_cast<int>(this->_ind));
			ret._ind = this->_ind - i;
			return ret;
		}

		double operator*() const;

		friend bool operator!=(const SequenceIterator&, const SequenceIterator&);

	private:

		Index			_ind;
		const Sequence*	_p_seq;
	};
}

inline bool UtilLib::operator!=(const UtilLib::SequenceIterator& it1, const UtilLib::SequenceIterator& it2){
	assert(it1._p_seq == it2._p_seq);
	return (it1._ind != it2._ind); 
}

inline bool UtilLib::operator==(const UtilLib::SequenceIterator& it1, const UtilLib::SequenceIterator& it2){
	return ! operator!=(it1,it2);
}


#endif // include guard
