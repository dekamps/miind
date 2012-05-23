// Copyright (c) 2005 - 2008 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_LAYERORDER_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LAYERORDER_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <cassert>
#include <vector>
#include "../UtilLib/UtilLib.h"

using UtilLib::Number;
using NetLib::NodeId;

// Created:				23-07-1999
// Author:				Marc de Kamps


namespace NetLib {


	//! NodeIterator
	//! NodeIterator is an iterator which can be used by node container classes
	template <class NodeType>
	class NodeIterator
	{
	public:

		//! Typically uses by the container that offers iterator services
		NodeIterator(NodeType*);

		//! Copy constructor (typically used by clients from containers, fro example in begin()
		NodeIterator( const NodeIterator<NodeType>& );

		//! Provide 'jumping ahead'
		NodeIterator  operator+(int);

		//! increment operator
		NodeIterator& operator++();

		//! increment operator
		NodeIterator  operator++(int);

		//! dereference operator
		NodeType* operator->() const; 

	private:

		NodeType*    _p_current_node;

	}; // end of LayerOrder

	template  <class NodeType>
	bool operator!=( const NodeIterator<NodeType>&, const NodeIterator<NodeType>& );

	typedef NodeIterator<double> D_NodeIterator;
	
} // end of Connectionism

#endif //include guard
