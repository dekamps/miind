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

/*! \file Contains the SparseNodeCreationExample */

#include "SparseNodeCreationExample.h"
#include "SparseNodeCode.h"

using namespace SparseImplementationLib;
using namespace NetLib;

//! SparseNodeCreationExample
bool SparseImplementationLib::SparseNodeCreationExample()
{
  // Example code for connecting nodes to each other.
  // Illustration purposes only ! Don't code like this !
  // Use SparseImplementation.

	D_SparseNode node_1, node_2, node_3;

	// set Ids
	node_1.SetId(NodeId(1));
	node_2.SetId(NodeId(2));
	node_3.SetId(NodeId(3));

	// create two connections: 2->1 and 3->1
	pair<D_SparseNode*,double> connection_12(&node_2, 2.0);

	pair<D_SparseNode*,double> connection_13(&node_3,-2.0);

	// add the Connections
	node_1.PushBackConnection(connection_12);
	node_1.PushBackConnection(connection_13);

	// Set activation in Node 2 and 3
	node_2.SetValue(1.0);
	node_3.SetValue(1.0);

	if ( node_1.InnerProduct() != 0 ) 
	  // this would be an error
		return false;

	// We can give a node a squashing function:
	Sigmoid sigmoid;
	node_1.ExchangeSquashingFunction(&sigmoid);
	return true;
}
