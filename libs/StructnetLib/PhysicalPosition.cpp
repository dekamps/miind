// Copyright (c) 2005 - 2008 Marc de Kamps
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

#include "PhysicalPosition.h"
#include "../NetLib/NetLib.h"

using namespace std;
using namespace StructnetLib;
using namespace NetLib;

ostream& StructnetLib::operator<<(ostream& s, const PhysicalPosition& position )
{
	s << position._position_depth << "\t" 
	  << position._position_x     << "\t" 
	  << position._position_y     << "\t" 
	  << position._position_z;

	return s;
}

istream& StructnetLib::operator>>(istream& s, PhysicalPosition& position)
{
	s >> position._position_depth 
	  >> position._position_x 
	  >> position._position_y 
	  >> position._position_z;

	return s;
}

bool StructnetLib::operator==( const PhysicalPosition& a , const PhysicalPosition& b )
{
	if ( ( a._position_x != b._position_x ) || 
	     ( a._position_y != b._position_y ) ||
		 ( a._position_z != b._position_z ) ||
		 ( a._position_depth != b._position_depth ) )
		return false;
	else
		return true;
}

bool operator< ( const PhysicalPosition&, const PhysicalPosition& )
{
	throw UnimplementedException(string("This function is not implemented"));
}

