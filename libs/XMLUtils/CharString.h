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
//	Implementing file:		CharString.cpp
//	Authors:			Johannes Drever, Melanie Dietz
//	Creation date:			December 2005
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

#ifndef _CODE_LIBS_XMLUTIL_CHARSTRING_H_
#define _CODE_LIBS_XMLUTIL_CHARSTRING_H_

//! \file CharString.h

#include <ostream>
#include <string>
#include <xercesc/util/XMLString.hpp>

namespace XMLUtils
{

  /*! \class CharString
   * \brief Utility class for char based strings.
   */
  class CharString
    {
    public:
      //! \brief Default constructor.
      CharString();

      /*! \brief Converting constructor.
       * \param str unicode string to be copied
       */
      CharString( XMLCh* str );
      
      /*! \brief Converting constructor.
       * \param str unicode string to be copied (const)
       */
      CharString( const XMLCh* str );
      
      /*! \brief Destructor, releases internal stored char string.
       */
      virtual ~CharString();
      
      /*! \brief Extracts the path part from a file URL. 
       * \param filename the file name with integrated path
       */
      static std::string getPath( const std::string& filename );

      /*! \brief returns the number of characters in the stored string
       */
      int getSize() const;

      /*! \brief returns a copy to the stored string
       */
      std::string getString() const;

      /*! \brief Determines the path type.
       * Whether the given url is an absolute path (true) or a relative path (false).
      */
      static bool isAbsolutePath( const std::string& path );

      /*! \brief Returns the pointer to the internal stored char string.
       */
      operator std::string() const;

      /*! \brief Returns a requested character out of the stored string.
       * \param i the index of the requested character.
       */
      char& operator[] ( const int i );

      /*! \brief Returns a requested character out of the stored string.
       * \param i the index of the requested character.
       */
      const char operator[] ( const int i ) const;

    protected:

    private:
      char* str; //!< the first address of the stored character string
    };

  /*! \fn std::ostream& operator<<( std::ostream& stream, const CharString& charString )
   * \brief Streaming operator for CharString data.
   * \param stream the target to write to
   * \param charString what to write to the stream
   */
  std::ostream& operator<<( std::ostream& stream, const CharString& charString );
}

#endif // _CODE_LIBS_XMLUTIL_CHARSTRING_H_
