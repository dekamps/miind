// Copyright (c) 2005, 2006 Melanie Dietz, Johannes Drever, Marc de Kamps
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

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
//	Declaring file: 		CharString.h
//	Authors:			Johannes Drever, Melanie Dietz
//	Creation date:			December 2005
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#include "CharString.h"

using XERCES_CPP_NAMESPACE_QUALIFIER XMLString;

//! \file CharString.cpp

namespace XMLUtils
{
  CharString::CharString() :
    str( NULL )
  {}

  CharString::CharString( XMLCh* str ) :
    str( NULL )
  {
    this->str = XMLString::transcode( str );
  }

  CharString::CharString( const XMLCh* str ) :
    str( NULL )
  {
    this->str = XMLString::transcode( str );
  }

  CharString::~CharString()
  {
    if ( NULL != this->str )
      {
	XMLString::release( &this->str );
      }
  }

  std::string
  CharString::getPath( const std::string& filename )
  {
    std::string::size_type pathPos = filename.find_last_of( "/" );

    std::string path;

    if ( std::string::npos != pathPos )
      {
	path = filename.substr( 0, pathPos ).c_str();
      }
    else
      {
	path = ".";
      }

    return path;
  }

  int
  CharString::getSize() const
  {
    return XMLString::stringLen( this->str );
  }

  std::string
  CharString::getString() const
  {
    return this->str;
  }

  bool
  CharString::isAbsolutePath( const std::string& path )
  {
    if ( ( path.substr( 0, 1 ) == "/" ) ||
	 ( path.substr( 0, 1 ) == "~" ) ||
	 ( path.substr( 1, 2 ) == ":/" ) ||
	 ( path.substr( 0, 7 ) == "file://" ) )
      { // absolute
	return true;
      }
    else
      { // relative
	return false;
      }
  }

  CharString::operator std::string() const
  {
    return this->str;
  };

  char&
  CharString::operator[] ( const int i )
  {
    return this->str[ i ];
  }

  const char
  CharString::operator[] ( const int i ) const
  {
    return this->str[ i ];
  }

  std::ostream&
  operator<<( std::ostream& stream, const CharString& charString )
  {
    stream << charString.getString();
    return stream;
  };
}
