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
//	Declaring file: 		XMLReader.h
//	Authors:			Johannes Drever, Melanie Dietz
//	Creation date:			02. January 2006
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

#include "XMLReader.h"

//! \file XMLReaderCode.h

//#define DEBUG

using namespace LayerMappingImplementationLib;

using namespace XMLUtils;


//----------------------------
template <class DataType>
XMLReader<DataType>::XMLReader
( AbstractNetworkBuilder<DataType>* pBuilder, 
  std::string filename ):
  AbstractReader<DataType>( pBuilder ),
  _filename( filename )
{
}

template <class DataType>
XMLReader<DataType>::~XMLReader( )
{
}


//----------------------------
template <class DataType>
AbstractNetwork<DataType>* XMLReader<DataType>::createNetwork( )
{
  // create a DOM document and a handler
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument * myDoc = NULL;
  XMLHandler* handler = XMLHandler::getInstance();
  if( (myDoc = handler->getDOM( _filename.c_str())) == 0 )
    throw LMImplBasicException( "Unable to open configuration XML file!" );
	
  // get all LMI from this file and instantiate them
  XStr rootName( STR_LAYERMAPPINGNETWORK_HEADER );
	
  // get all LMI from this file and instantiate them
  XStr lmiName( STR_LAYER_HEADER );
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNodeList* list_layers = 
    myDoc->getElementsByTagName( lmiName.unicodeForm() );

  for( unsigned int i=0; i< list_layers->getLength(); ++i ) {
    _parseLayerMappingImplementation( myDoc, list_layers->item(i) );
  }		

  handler->releaseDOM( myDoc );

  return AbstractReader<DataType>::_pBuilder->getNetwork( );
}


//--------------------------------
template <class DataType>
bool XMLReader<DataType>::_parseLayerMappingImplementation
( XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *myDoc, 
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* rootNode )
{
  // rootNode is pointing to node <LMI>
  XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker * walker = 
    myDoc->createTreeWalker( rootNode, 
			     XERCES_CPP_NAMESPACE_QUALIFIER DOMNodeFilter::SHOW_ELEMENT, 
			     new XMLFilter(), 
			     false );
  try
    {
      // pCurrentNode is either InputLayer, OutputLayer, Assignement or URFF
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pCurrentNode = walker->nextNode();

      while ( pCurrentNode != NULL ) {
	const XMLCh* pChNodeName = pCurrentNode->getNodeName();
	if( pChNodeName != NULL ) {
	  string currentNodeName	= CharString( pChNodeName );
#ifdef DEBUG
	  cout << "[DEBUG]\tXMLReader::parseLMI\t Processing node: " << currentNodeName << endl;
#endif
					
	  if( currentNodeName == STR_LAYERDESCRIPTION_INLAYER_HEADER )
	    _parseInputLayer( walker );
	  else if ( currentNodeName == STR_LAYERDESCRIPTION_OUTLAYER_HEADER )
	    _parseOutputLayer( walker );
	  else if ( currentNodeName == STR_SLA_SUBLAYERASSIGNEMENT_HEADER )
	    _parseAssignments( walker );
	  else if ( currentNodeName == STR_FUNCTIONS_HEADER )
	    _parseFunctions( walker );
	}
	pCurrentNode = walker->nextSibling();
      }
      walker->release();
      AbstractReader<DataType>::_pBuilder->addLMI( );
	
    } catch ( ... ) 
      { 
#ifdef DEBUG
	cout << endl << "!!![ERROR]!!!\t"; 
#endif
      }
  return true;
}
//------------------------------------
template <class DataType>
bool XMLReader<DataType>::_parseInputLayer
( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker)
{
#ifdef DEBUG
  cout << "[DEBUG]\t XMLReader::parseInputLayer\t" << endl;
#endif
  map< string, string > properties;
  _parseLayerDescription( properties, walker);
	
#ifdef DEBUG
  _printMaps( properties );
#endif
	
  AbstractReader<DataType>::_pBuilder->setInputLayerDescription( properties );
	
  return true;
}

//-----------------------------------------------
template <class DataType>
bool XMLReader<DataType>::_parseOutputLayer
( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker)
{
#ifdef DEBUG
  cout << "[DEBUG]\t XMLReader::parseOutputLayer\t" << endl;
#endif
  map< string, string > properties;
  _parseLayerDescription( properties, walker);
	
#ifdef DEBUG
  _printMaps( properties );
#endif
	
  AbstractReader<DataType>::_pBuilder->setOutputLayerDescription( properties );
  return true;
}

//-----------------------------------------------
template <class DataType>
bool XMLReader<DataType>::_parseFunctions
( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker)
{
#ifdef DEBUG
  cout << "[DEBUG]\t XMLReader::parseFunctions\t" << endl;
#endif
	
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pEntryNode = walker->getCurrentNode();
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pFunction = walker->firstChild();
	
	
	
  while( pFunction != NULL && 0 != 
	 (pFunction->compareTreePosition(pEntryNode) & 
	  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::TREE_POSITION_ANCESTOR) )
    {
      map <string, string > properties;
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pChild = walker->firstChild();
	
      string type = CharString( pFunction->getNodeName() );

      while( pChild != NULL && 0 != 
	     (pChild->compareTreePosition(pFunction)
	      & XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::TREE_POSITION_ANCESTOR) )
	{
	  string key = CharString( pChild->getNodeName() );
	  string value = CharString( pChild->getFirstChild()->getNodeValue() );
	  if ( value.length() > 0 )
	    properties[key] = value ;		
	  pChild = walker->nextNode();
	}
      
#ifdef DEBUG
      _printMaps( properties );
#endif
      
      AbstractReader<DataType>::_pBuilder->addFunction( type, properties );
      if( pChild != NULL )
	pFunction = pChild;
      else
	pFunction = walker->nextNode();
    }
	
  walker->setCurrentNode(pEntryNode);
  return true;
}

//-----------------------------------------------
template <class DataType>
bool XMLReader<DataType>::_parseAssignments
( XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker)
{
#ifdef DEBUG
  cout << "[DEBUG]\t XMLReader::parseAssignements\t" << endl;
#endif
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pEntryNode = walker->getCurrentNode();
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pChild = walker->firstChild();
	
  map <string, string > properties;
	
  while( pChild != NULL && 0 != 
	 (pChild->compareTreePosition(pEntryNode) & 
	  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::TREE_POSITION_ANCESTOR) )
    {
      string key = CharString( pChild->getNodeName() );
      string value = CharString( pChild->getFirstChild()->getNodeValue() );
      if ( value.length() > 0 ) {
	properties[key] = value;
#ifdef DEBUG
	_printMaps( properties );
#endif
	AbstractReader<DataType>::_pBuilder->addAssignement( properties );
      }
      pChild = walker->nextNode();
    }

  walker->setCurrentNode(pEntryNode);
  return true;
}

//-----------------------------------------------
template <class DataType>
void XMLReader<DataType>::_printMaps( map <string, string>& properties ) const
{
  for( map< string, string >::iterator it = properties.begin();
       it != properties.end(); ++it ) 
    {
      cout << "[DEBUG]\t XMLReader::map\t" << it->first 
	   << "\t" << it->second << endl;
    }
}

//-----------------------------------------------
template <class DataType>
void XMLReader<DataType>::_parseLayerDescription
( map <string, string>& properties, 
  XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker )
{
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pEntryNode = walker->getCurrentNode();
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* pChild = walker->firstChild();
	
  enum SKIP { UNDEFINED, RF_X, RF_Y};
  short type = UNDEFINED;

  while( pChild != NULL && 0 != 
	 (pChild->compareTreePosition(pEntryNode) & 
	  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::TREE_POSITION_ANCESTOR) )
    {
      string key = CharString( pChild->getNodeName() );
      if( key == STR_LAYERDESCRIPTION_RF_SKIPS_X_RATIONAL ) {
	type = RF_X;
	pChild = walker->nextNode();
	continue;
      }
      else if( key == STR_LAYERDESCRIPTION_RF_SKIPS_Y_RATIONAL ) {
	type = RF_Y;
	pChild = walker->nextNode();
	continue;
      }
      else if( key == STR_LAYERDESCRIPTION_RF_SKIPS_DENOMINATOR ) {
	if( type == RF_X )
	  key = STR_LAYERDESCRIPTION_RF_SKIPS_X_RATIONAL_DENOMINATOR;
	else if( type == RF_Y )
	  key = STR_LAYERDESCRIPTION_RF_SKIPS_Y_RATIONAL_DENOMINATOR;
      }
      else if( key == STR_LAYERDESCRIPTION_RF_SKIPS_NUMERATOR ) {
	if( type == RF_X )
	  key = STR_LAYERDESCRIPTION_RF_SKIPS_X_RATIONAL_NUMERATOR;
	else if( type == RF_Y )
	  key = STR_LAYERDESCRIPTION_RF_SKIPS_Y_RATIONAL_NUMERATOR;
      }
		
      string value = CharString( pChild->getFirstChild()->getNodeValue() );
      if ( value.length() > 0 ) {
	properties[key] = value ;
      }
      pChild = walker->nextNode();
    }
	
  walker->setCurrentNode(pEntryNode);
}
//-----------------------------------------------
//made copy&paste from Rational.cpp; was not tested !!
template <class DataType>
void XMLReader<DataType>::_parseRational
( Rational &value, XERCES_CPP_NAMESPACE_QUALIFIER DOMTreeWalker *walker )
{
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode* rootNode = walker->getCurrentNode();
  XERCES_CPP_NAMESPACE_QUALIFIER DOMNode * currentNode = walker->nextNode();	

  while ( ( currentNode != NULL ) && currentNode->getParentNode()->isSameNode( rootNode ) )
    {
      string currentNodeName = CharString( currentNode->getNodeName() );
		
      if( currentNodeName == "numerator" )
	{
	  stringstream ss ( stringstream::in | stringstream::out );
	  ss.str( CharString( currentNode->getFirstChild()->getNodeValue() ) );
	  ss >> value._numerator;
	}
      else if( currentNodeName == "denominator" )
	{
	  stringstream ss ( stringstream::in | stringstream::out );
	  ss.str( CharString( currentNode->getFirstChild()->getNodeValue() ) );
	  ss >> value._denominator;
	}
      currentNode = walker->nextNode();
    }
		
  walker->previousNode();
}
