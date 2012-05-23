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

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
//	Implementing file:		XMLReaderCode.h
//	Authors:			Melanie Dietz
//	Creation date:			02. January 2006
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

/*! \file XMLReader.h */
#ifndef _LMI_LIB_XMLUTIL_XMLREADER_INCLUDE_GUARD
#define _LMI_LIB_XMLUTIL_XMLREADER_INCLUDE_GUARD

#include "../LayerMappingImplementationLib/AbstractReaderCode.h"
#include "../LayerMappingImplementationLib/AbstractNetworkBuilderCode.h"
#include "XMLHandler.h"
#include "XMLFilter.h"
#include "CharString.h"

#include <xercesc/dom/DOM.hpp>

using namespace LayerMappingImplementationLib;


namespace XMLUtils {
  
  /*! \class XMLReader
    \brief Gets data from XML files and calles LMI lib network builders for network creation.
  */
  template <class DataType>
    class XMLReader :
    public AbstractReader<DataType>
    {
      
    public:
      /*! \brief Default constructor
       * \param pBuilder the network builder to be used
       * \param filename the source for network parameters
       */
      XMLReader( AbstractNetworkBuilder<DataType>* pBuilder, std::string filename );

      //! \brief Destructor.
      virtual ~XMLReader( );
      
      /*! \brief Returns the generated network.
       * The user is responsible for deleting the returned pointer!
       */
      virtual AbstractNetwork<DataType>* createNetwork( );
      
    private:
      std::string			_filename; //!< the path to the source file

      //! \brief Creates layer mapping.
      bool _parseLayerMappingImplementation
	( XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *document,
	  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* rootNode );

      //! \brief Creates input layer description.
      bool _parseInputLayer
	( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker);
      
      //! \brief Creates output layer description.
      bool _parseOutputLayer
	( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker);
      
      //! \brief Instantiates mapping functions.
      bool _parseFunctions
	( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker);
      
      //! \brief Collects assignments of mapping functions and sublayer indizes.
      bool _parseAssignments
	( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker);

      //! \brief Streams the given key/value map to the standard output.
      void _printMaps( map <string, string>& properties ) const;
      
      //! \brief Low level parsing for layer descriptions.
      void _parseLayerDescription
	( map<string, string>& properties, 
	  XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker);

      //! \brief Low level parsing for rational numbers.
      void _parseRational
	( Rational& value, XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker );
    };

}

#endif
