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
//	Implementing file:		XmlString.cpp
//	Authors:			Johannes Drever, Melanie Dietz
//	Creation date:			December 2005
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#ifndef _CODE_LIBS_XMLUTIL_XMLSTRING_H_
#define _CODE_LIBS_XMLUTIL_XMLSTRING_H_

//! \file XmlString.h

#include <string>
#include <xercesc/dom/DOM.hpp>

using std::string;

namespace XMLUtils
{
  /*! \class XmlString
   * \brief Converter class for strings and XMLCh.
   * \sa class CharString
   */
  class XmlString
    {
    public:
      //! \brief Default constructor.
      XmlString();

      /*! \brief Converting constructor.
	\param str the string to be converted
       */
      XmlString( const std::string& str );

      //! \brief Default destructor.
      virtual ~XmlString();

      //! \brief Returns number of characters stored.
      int getSize() const;

      //! \brief Returns unicode converted string.
      XMLCh* getString() const;

      //! \brief Returns pointer to first character of stored XMLCh array.
      operator const XMLCh*() const;

      /*! \brief Grants access to stored char array.
       * \param i index which character is accessed.
       */
      XMLCh& operator[] ( const int i );

      /*! \brief Returns content of character array.
	\param i index which character is returned.
       */
      const XMLCh operator[] ( const int i ) const;

    protected:

    private:
      XMLCh* str; //!< internal stored XMLCh* pointer.
    };
}

#endif // _CODE_LIBS_XMLUTIL_XMLSTRING_H_
