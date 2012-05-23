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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_UTILLLIB_ITERATORITERATOR_INCLUDE_GUARD
#define _CODE_LIBS_UTILLLIB_ITERATORITERATOR_INCLUDE_GUARD

#include <utility>
#include <iterator>
#include <boost/shared_ptr.hpp>
#include "Sequence.h"

using boost::shared_ptr;
using std::forward_iterator_tag;
using std::iterator;
using std::pair;

namespace UtilLib {

	typedef vector<double> ParameterValues;

	class SequenceIteratorIterator : public iterator<forward_iterator_tag,ParameterValues>{
	public:

		SequenceIteratorIterator(bool);

		bool AddLoop(const Sequence&);
	
		//! Define the customary iterator increment (prefix)	
		SequenceIteratorIterator& operator++();

		//! Define the customary iterator increment (postfix)	
		SequenceIteratorIterator operator++(int);

		ParameterValues& operator*(); 

		ParameterValues* operator->();

		vector<string> NamesList();

		string CurrentName();

		Number size() const;

		SequenceIteratorIterator& operator+(Index);

		friend bool operator!=
		(
			const SequenceIteratorIterator&, 
			const SequenceIteratorIterator&
		);

	private:

		typedef pair<boost::shared_ptr<Sequence>,Sequence::iterator> sequence_with_iterator;

		void	ConfigureLoop		();
		void	ShiftIndices		();
		void	CopyToPlaceHolder	();

		bool							_is_unique;
		bool							_has_not_started;
		Index							_ind;
		mutable  Number					_size; // mutable because only calculated when all loops are added

		vector<sequence_with_iterator>	_vec_sequences;
		vector<double>					_vec_placeholder;
	};

	SequenceIteratorIterator operator+(const SequenceIteratorIterator&, Index);

	bool operator!=(const SequenceIteratorIterator&, const SequenceIteratorIterator&);
}

#endif // include guard
