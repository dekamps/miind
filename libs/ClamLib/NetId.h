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
#ifndef _CODE_LIBS_CLAMLIB_NETID_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_NETID_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>

using std::ostream;
using std::istream;

#include <TNamed.h>

//namespace ClamLib {

//! NodeId
//! Author: Marc de Kamps
//! Date:   04-06-2008
//!
//! This is a replication of NetLib::NodeId, but compliant with ROOT serialization
//! At a later stage we will investigate if this class can be eliminated (TODO)

	struct NetId : public TObject
	{
		//! Default constructor
		explicit NetId():_id_value(0){}

		//! Explict construction
		explicit NetId(int id_value ):_id_value(id_value){}

		//! copy construction
		NetId(const NetId&);

		NetId& operator=(const NetId&);

		//! id value, negative values are invalid
		int _id_value;

//		ClassDef(NetId,1);

	}; // end of NetId
	
	//! explicit conversion function
	inline NetId ConvertToNetId(int i_value){return NetId(i_value);} 

	istream& operator>> ( istream&, NetId );
	ostream& operator<< ( ostream&, NetId );
	 bool    operator!= ( NetId, NetId );
	 bool    operator== ( NetId, NetId );
	 bool    operator<  ( NetId, NetId );

//} // end of ClamLib

#endif // include guard
