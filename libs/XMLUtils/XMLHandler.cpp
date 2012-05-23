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

#include "XMLHandler.h"

using XMLUtils::UtilDOMErrorHandler;
using XMLUtils::XMLHandler;
using UtilLib::UtilException;
using XERCES_CPP_NAMESPACE_QUALIFIER XMLString;
using XERCES_CPP_NAMESPACE_QUALIFIER XMLUni;

using std::ostream;
using std::cerr;
using std::string;

//---static variable initialization-----
XMLHandler* XMLHandler::_xmlhandler = NULL;


//---getInstance()-----
XMLHandler* XMLHandler::getInstance(ostream& ostr) 
{

  if( _xmlhandler != NULL ) {
    return _xmlhandler;
  }
  else {
    try {
      XERCES_CPP_NAMESPACE_QUALIFIER XMLPlatformUtils::Initialize();
      _xmlhandler = new XMLHandler(ostr);
      return _xmlhandler;
    }
    catch(const XERCES_CPP_NAMESPACE_QUALIFIER XMLException& toCatch)
      {
	char *pMsg = XMLString::transcode(toCatch.getMessage());
	ostr << "Error during Xerces-c Initialization.\n"
	     << "  Exception message:"
	     << pMsg;
	XMLString::release(&pMsg);
	return NULL;
      }
    catch( const XERCES_CPP_NAMESPACE_QUALIFIER DOMException& toCatch)
      {
	char *pMsg = XMLString::transcode(toCatch.msg);
	ostr << "Error during Xerces-c Initialization.\n"
	     << "  Exception message:"
	     << pMsg;
	XMLString::release(&pMsg);
	return NULL;
      }
  }
}

//---Private constructor()-----
XMLHandler::XMLHandler( ostream& ostr ):
  _out( ostr ),
  _parser( NULL ),
  _errorHandler( new UtilDOMErrorHandler() ),
  _impl( NULL )
{
  _impl = XERCES_CPP_NAMESPACE_QUALIFIER DOMImplementationRegistry::getDOMImplementation
    ( X( "Core" ) );
  
  _parser = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMImplementationLS*>
    ( _impl )->createDOMBuilder
    ( XERCES_CPP_NAMESPACE_QUALIFIER DOMImplementationLS::MODE_SYNCHRONOUS, 0 );
  
  if (_parser->canSetFeature(XMLUni::fgDOMValidation, true))
    _parser->setFeature(XMLUni::fgDOMValidation, true);
  if (_parser->canSetFeature(XMLUni::fgDOMNamespaces, true))
    _parser->setFeature(XMLUni::fgDOMNamespaces, true);
  if (_parser->canSetFeature(XMLUni::fgDOMDatatypeNormalization, true))
    _parser->setFeature(XMLUni::fgDOMDatatypeNormalization, true);
  if (_parser->canSetFeature(XMLUni::fgXercesSchema, true))
    _parser->setFeature(XMLUni::fgXercesSchema, true);
  if (_parser->canSetFeature(XMLUni::fgXercesSchemaFullChecking, true))
    _parser->setFeature(XMLUni::fgXercesSchemaFullChecking, true);
  if (_parser->canSetFeature(XMLUni::fgXercesUserAdoptsDOMDocument, true))
    _parser->setFeature(XMLUni::fgXercesUserAdoptsDOMDocument, true);
  if (_parser->canSetFeature(XMLUni::fgXercesValidationErrorAsFatal, true))
    _parser->setFeature(XMLUni::fgXercesValidationErrorAsFatal, true);
  if (_parser->canSetFeature(XMLUni::fgDOMWhitespaceInElementContent, true))
    _parser->setFeature(XMLUni::fgDOMWhitespaceInElementContent, true);

  _parser->setErrorHandler(_errorHandler);
}

//---Destructor()----
XMLHandler::~XMLHandler(void)
{
  _xmlhandler = NULL;
  _parser->release();
  delete _errorHandler;
  XERCES_CPP_NAMESPACE_QUALIFIER XMLPlatformUtils::Terminate();
}


//---getDOM()-----
XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument* XMLHandler::getDOM(const char* filename) 
{
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *doc = 0;
  try {
    doc = _parser->parseURI(string(filename).c_str());
  }
  catch (const XERCES_CPP_NAMESPACE_QUALIFIER XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "XMLHandler:\t";
    std::cout << message << std::endl;
    XMLString::release(&message);
    releaseDOM( doc );
    doc = 0;
    return NULL;
  }
  catch (const XERCES_CPP_NAMESPACE_QUALIFIER DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);
    std::cout << "XMLHandler:\t";
    std::cout << message << std::endl;
    XMLString::release(&message);
    doc = 0;
    return NULL;
  }
  catch (const XERCES_CPP_NAMESPACE_QUALIFIER SAXException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "XMLHandler:\t";
    std::cout << message << std::endl;
    //std::cout << "Exception message is: \n\t\t" << message << "\n";
    XMLString::release(&message);
    doc = 0;
    return NULL;
  }
  catch (const UtilException& toCatch) {
    std::cout << "XMLHandler:\t";
    std::cout << toCatch.what() << std::endl;
    //std::cout << "Exception message is: \n" << toCatch.Description() << "\n";
    doc = 0;
    return NULL;
  }
  catch (...) {
    std::cout << "XMLHandler:\t";
    std::cout << "Unexpected Exception \n" ;
    doc = 0;
    return NULL;
  }
  return doc;
}

//----releaseDOM()---
void XMLHandler::releaseDOM( XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument* doc)
{
  doc->release();
}
