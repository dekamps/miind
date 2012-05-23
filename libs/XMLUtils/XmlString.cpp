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
//	Declaring file:		        XmlString.h
//	Authors:			Johannes Drever, Melanie Dietz
//	Creation date:			December 2005
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#include "XmlString.h"

//! \file XmlString.cpp

using XERCES_CPP_NAMESPACE_QUALIFIER XMLString;
using XMLUtils::XmlString;

XmlString::XmlString() :
  str( 0L )
{}

XmlString::XmlString( const std::string& str ) :
  str( NULL )
{
  this->str = XMLString::transcode( str.c_str() );
}

XmlString::~XmlString()
{
  if ( NULL != this->str )
    {
      XMLString::release( &this->str );
    }
}

int
XmlString::getSize() const
{
  return XMLString::stringLen( this->str );
}

XMLCh*
XmlString::getString() const
{
  return this->str;
}

XmlString::operator const XMLCh*() const
{
  return this->str;
};

XMLCh&
XmlString::operator[] ( const int i )
{
  return this->str[ i ];
}

const XMLCh
XmlString::operator[] ( const int i ) const
{
  return this->str[ i ];
}
