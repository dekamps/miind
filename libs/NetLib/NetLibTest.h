
// Copyright (c) 2005 - 2007 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_NETLIBTEST_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_NETLIBTEST_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <vector>
#include <iostream>
#include <string>
#include <utility>
#include "../UtilLib/UtilLib.h"
#include "NodeLinkCollection.h"
#include "LayeredArchitecture.h"
#include "PatternCode.h"

using std::vector;
using std::string;
using std::ostream;
using std::pair;

using UtilLib::LogStream;

namespace NetLib
{
	//! Test class for the NetLib library
	class  NetLibTest : public LogStream
	{
	public:

		NetLibTest(); 
		NetLibTest
		(
				boost::shared_ptr<ostream>      // provide a stream for the log file
		); 
		bool Execute();

		LayeredArchitecture 
			CreateArchitecture  (bool b_threshold) const;

	private:

		// sub tests:
		bool LayerIteratorTest() const;
		bool ArchitectureTest () const;
		bool PatternTest      () const;

		const string _path_name;

	}; // end of NetLibTest
		
} // end of NetLib

#endif // include guard

