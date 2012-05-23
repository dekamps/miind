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
#ifndef _CODE_LIBS_NETLIB_NODELINKCOLLECTIONFROMSPARSEIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_NODELINKCOLLECTIONFROMSPARSEIMPLEMENTATION_INCLUDE_GUARD


#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <vector>
#include "../NetLib/NetLib.h"
#include "SparseImplementationAllocator.h"

using NetLib::NodeLink;
using NetLib::NodeLinkCollection;

// Name:   NodeLinkCollectionFromSparseImplementation
// Author: Marc de Kamps
// Date:   05-08-2003
// Short description: Auxilliary object, the CreateReverseNodeLinkCollection is instrumental 
// in generating reversed SparseImplementations

using std::vector;

namespace SparseImplementationLib
{
	//! This is an auxilliary class which depends on implementation details of SparseImplementation
	//! It takes the vector which is used inside a SparseImplementation to store the nodes and is able
	//! to convert it into a NodeLinkCollection. It is also able to generate the reverse network and that
	//! is the main reason for its existence. This class is used to generate the reverse connections in
	//! ReverseSparseNode, which is necessary, for example, in Backpropagation.
	template <class NodeType>
	class NodeLinkCollectionFromSparseImplementation
	{
	public:
		
		//! constructor
		NodeLinkCollectionFromSparseImplementation();

		//! takes the vector of nodes inside a SparseImplementation and generates the NodeLinkCollection
		//! necessary to create the network represented in SparseImplementation.
		NodeLinkCollection CreateNodeLinkCollection       
		(
			const vector<NodeType, SparseImplementationAllocator<NodeType> >&
		);

		//! generates the NodeLinkCollection of the reverse network, i.e. input-output relations are inverted
		//! with respect to the method above.
		NodeLinkCollection CreateReverseNodeLinkCollection
		(
			const vector<NodeType, SparseImplementationAllocator<NodeType> >&
		);

	private:
			
		vector<NodeLink> CreateNodeLinkVector
		(
			const vector<NodeType, SparseImplementationAllocator<NodeType> >&
		);  // auxilliary routine 
		void  AssignReversedRelation  
		(
			vector<NodeLink>&, 
			const vector<NodeType, SparseImplementationAllocator<NodeType> >&
		);

	}; // end of NodeLinkCollectionFromSparseImplementation

} // end of NetLib

#endif // include guard
