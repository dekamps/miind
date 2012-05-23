// Copyright (c) 2005 - 2011 Marc de Kamps
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
#ifndef _CODE_LIBS_MIINDLIB_XMLNODES_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_XMLNODES_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "../UtilLib/UtilLib.h"

using DynamicLib::NodeType;

namespace MiindLib {

	//! Converts a NodeType into the right XML value for an XMLNode tag
	string FromTypeToValue(DynamicLib::NodeType);

	//! Converts a type value attribute into a NodeType
	DynamicLib::NodeType FromValueToType(const string&);

	//! XML counterpart of a DynamicNode for use in a SimulationBuilder
	class XMLNode  :public Persistant {
	public:

		XMLNode(istream&);

		//! XML counterpart of a DynamicNode
		XMLNode
		(
			const string&,	//!< type of the DynamicNode
			const string&,	//!< name of the DynamicNode for referencing in the XML file
			const string&	//!< name of the Algorithm to be associated with this node
		);

		XMLNode(const XMLNode&);

		virtual ~XMLNode();

		virtual string Tag() const;

		virtual bool ToStream(ostream&) const;

		virtual bool FromStream(istream&);


		string AlgorithmName() const { return _alg_name; }

		string TypeName () const { return _type_name; }

	private:

		string		_type_name;
		string		_alg_name;
	
	};

	typedef vector<XMLNode> node_vector;
}
#endif // include guard