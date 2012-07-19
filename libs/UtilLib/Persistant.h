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
#ifndef _CODE_LIBS_UTILLIB_PERSISTANT_INCLUDE_GUARD
#define _CODE_LIBS_UTILLIB_PERSISTANT_INCLUDE_GUARD

#include <string>
#include "Streamable.h"

using std::string;

namespace DynamicLib {
	template <class T> class AlgorithmBuilder;
}

namespace UtilLib {

	//! Objects of classes deriving from this class are deemed to be sufficiently complex that normal streaming operators should not be used.
	//! These classes should define a constructor which accepts an input stream as an argument.
	//! Subclasses can overloaded the IsValid function to indicate whether construction was successful. If the object is invalid
	//! the BuildFailureReason method will return the string which caused the building from input stream to fail
	class Persistant : public Streamable {
	public:

		//! A subclass may be constructed by other means than an input stream
		Persistant();

		//! Any deriving class must be able to accept an input stream
		Persistant(istream&);

		//! virtual destructor
		virtual ~Persistant() = 0;

		virtual bool IsValid() const {return _b_valid; }

		string BuildFailureReason() const { return _offender;}

		template <class T> friend class DynamicLib::AlgorithmBuilder;

	protected:

		void SetValidity(bool);

		void SetOffendingString(const string&);

	private:

		bool _b_valid;

		string _offender;
	};
}

#endif // include guard
