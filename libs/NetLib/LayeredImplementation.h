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
#ifndef _CODE_LIBS_NETLIB_LAYEREDIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LAYEREDIMPLEMENTATION_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include "../UtilLib/UtilLib.h"
#include "BasicDefinitions.h"
#include "LayeredArchitecture.h"

using std::string;
using std::istream;
using std::vector;
using UtilLib::Number;
using UtilLib::Streamable;

///////////////////////////////////////////////////////////////////////////////////////////
// Created:				27-09-1999
// Module:				ATTENT.NN.NETIMP
// Author:				Marc de Kamps
// Version:				0.00
// Short Description:	Base class for network implementations
///////////////////////////////////////////////////////////////////////////////////////////



namespace NetLib
{

	//! LayeredImplementation

	class LayeredImplementation : public Streamable
	{
	public:

		//! Read a LayeredImplementation from a stream
		LayeredImplementation(istream&);

		//! Create a LayeredImplementation from a LayeredArchitecture
		LayeredImplementation(const LayeredArchitecture&);

		LayeredImplementation(const LayeredImplementation&);
	    ~LayeredImplementation();

		LayeredImplementation &	operator=(const LayeredImplementation&);

		bool ToStream   (ostream&) const;
		bool FromStream (istream&);
		
		string Tag() const;

		// network properties:

		Number  NumberOfTotalNodes  () const;
		Number  SizeOfInputLayer    () const;
		Number  SizeOfOutputLayer   () const;

		//! Number of Layers
		Number	NumberOfLayers		() const;

		//! Number of nodes ina given layer
		Number	NumberOfNodesInLayer(Layer) const;  // number of Nodes in a given layer

		Number  NrConnections	        () const;
		Number  MaxNumberOfNodesInLayer () const;
		Number  NrConnectionFrom(size_t) const;

		NodeId  BeginId			(Layer) const;
		NodeId  EndId           (Layer) const;

		vector<Number> ArchVec	()		 const;
		
	protected:

	private: 

		// Auxilliary parsing functions for istream constructor:

		istream&       RemoveHeader             (istream&) const;
		istream&       RemoveFooter             (istream&) const;
		int            ParseNumberOfNodes       (istream&) const;
		vector<Number> ParseArchitectureVector  (istream&) const;
		vector<Number> ParseOriginatingVector   (istream&) const;
		vector<NodeId> ParseBeginIds            (istream&) const;

		Number _number_of_connections;
		Number _number_of_nodes;

		vector<Number>   _vector_architecture;  // Number neurons per layer
		vector<Number>   _vector_begin_connection;
		vector<NodeId>   _vector_begin_id;      // The first NodeId of each layer
		vector<Number>   _vector_originating;   // Number of connections originating from layer


		// declaration of the tags, used internally

		static string _tag_architecture;
		static string _tag_originating;
		static string _tag_beginid;

	}; // end of LayeredImplementation

	ostream& operator<<(ostream&, const LayeredImplementation&);

} // end of NetLib

#endif // include guard
