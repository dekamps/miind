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
#ifndef _CODE_LIBS_UTILLIB_SEQUENCE_INCLUDE_GUARD
#define _CODE_LIBS_UTILLIB_SEQUENCE_INCLUDE_GUARD

#include <iostream>
#include "BasicDefinitions.h"
#include "Streamable.h"
#include "SequenceIterator.h"



using std::istream;

namespace UtilLib {

	class Sequence;
	bool operator==(const Sequence&, const Sequence&);

	class Sequence : public Streamable {
	public:

		typedef SequenceIterator iterator;

		Sequence(){}
		
		Sequence(istream&);

		virtual ~Sequence() = 0;

		virtual double operator[](Index) const = 0;

		SequenceIterator begin() const;

		SequenceIterator end() const;

		Number size() const;

		virtual string Name() const = 0;

		virtual bool ToStream(ostream&) const;

		virtual bool FromStream(istream&);

		virtual string Tag() const = 0;
	
		virtual Sequence* Clone() const = 0;

		friend bool operator==(const Sequence&, const Sequence&);

	protected:

		virtual Index BeginIndex() const = 0;
		virtual Index EndIndex() const = 0;

	private:

	};
}

inline bool UtilLib::operator==(const Sequence& seq1, const Sequence& seq2)
{
	return (&seq1 == &seq2);
}

#endif // include guard

