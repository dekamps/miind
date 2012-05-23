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
//	Implementing file:		HMLHandler.cpp
//	Authors:			Melanie Dietz
//	Creation date:			December 2005
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#ifndef _CODE_LIBS_XMLUTIL_XMLHANDLER_H
#define _CODE_LIBS_XMLUTIL_XMLHANDLER_H

//! \file XMLHandler.h

#include <iostream>
#include <string>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>

#include "UtilDOMErrorHandler.h"

namespace XMLUtils
{
  /*! \class XStr
   * \brief Converting class for XMLChar data types.
   */
  class XStr
    {
      public :
	// -----------------------------------------------------------------------
	//  Constructors and Destructor
	// -----------------------------------------------------------------------

	/*! \brief Default constructor.
	 * \param toTranscode a character array to store as XMLCh array
	 */
	XStr(const char* const toTranscode)
	{
	  // Call the private transcoding method
	  fUnicodeForm = XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(toTranscode);
	}
      
      /*! \brief Default constructor.
	\param toTranscode a string object to store as XMLCh array
      */
      XStr(const std::string toTranscode)
	{
	  fUnicodeForm = XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(toTranscode.c_str());
	  //basic_string::c_str() is managed/owned by string class -> not to be modified or deleted!
	}
      
      //! \brief Default destructor that releases the stored XMLCh array.
      ~XStr()
	{
	  XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&fUnicodeForm);
	}
      

      // -----------------------------------------------------------------------
      //  Getter methods
      // -----------------------------------------------------------------------
      //! \brief Returns the unicode version of the stored string.
      const XMLCh* unicodeForm() const
	{
	  return fUnicodeForm;
	}
      
      private :
	// -----------------------------------------------------------------------
	//  Private data members
	//
	//  fUnicodeForm
	//      This is the Unicode XMLCh format of the string.
	// -----------------------------------------------------------------------
	XMLCh*   fUnicodeForm; //!< the stored string
    };
  
  /*! \def X(str) 
   * Calls XStr::unicodeForm with instantiating XStr with the parameter.
   */
#define X(str) XStr(str).unicodeForm()
  

  /*! \class XMLHandler 
   * \brief Controls access to DOMDocuments.
   */
  class XMLHandler 
    {
      
    public:
      /*! \brief Returns an instance of current XMLHandlers.
       */
      static XMLHandler* getInstance( std::ostream& ostr = std::cout );
      
      /*! \brief Returns a handler for the referenced XML document.
       * \param filename the XML document to be accessed (read/write)
       */
      XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument* getDOM(const char* filename);

      /*! \brief Saves changes to XML tree and releases XML document.
       * \param doc the DOM document to be released.
       */
      void releaseDOM( XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument* doc);
      
    private:
      /*! \brief Constructor that initializes the handler parameters.
       * Initializes the XMLPlatformUtils.
       * \param ostr target stream for logging
       */
      XMLHandler(std::ostream& ostr);
      
      //! \brief Default constructor. Terminates the XMLPlatformUtils.
      ~XMLHandler( );
      
      std::ostream&	_out; //!< ostream for all messages
      XERCES_CPP_NAMESPACE_QUALIFIER DOMBuilder* _parser; //!< the utilized DOMBuilder
      UtilDOMErrorHandler* _errorHandler; //!< the error handler
      XERCES_CPP_NAMESPACE_QUALIFIER DOMImplementation*	_impl; //!< the DOM implementation
      static XMLHandler* _xmlhandler;	//!< the single XMLHandler instance for the file
    };
}

#endif //_CODE_LIBS_UTIL_XMLHANDLER_H
