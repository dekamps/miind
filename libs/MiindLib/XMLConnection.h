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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_MIINDLIB_XMLCONNECTION_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_XMLCONNECTION_INCLUDE_GUARD

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/foreach.hpp>
#include "../UtilLib/UtilLib.h"
#include "MiindLibException.h"


using std::istream;
using std::ostream;
using std::istringstream;
using std::string;
using std::vector;

using UtilLib::AttributePair;
using UtilLib::AttributeList;

namespace MiindLib {

	template <class Weight>
	class XMLConnection : public Persistant {
	public:

		XMLConnection(istream&);

		XMLConnection
		(
			const string& name_in, 
			const string& name_out, 
			const Weight& weight
		):
		_name_in(name_in),
		_name_out(name_out),
		_weight(weight)
		{
		}

		virtual ~XMLConnection();

		virtual bool ToStream(ostream&) const;

		virtual bool FromStream(istream&);

		virtual string Tag() const;

		string In() const {return _name_in; }

		string Out() const { return _name_out; }

		Weight Connection() const {return _weight; }

	private:

		string _name_in;
		string _name_out;
		Weight _weight;
	};
}

#endif // inlcude guard