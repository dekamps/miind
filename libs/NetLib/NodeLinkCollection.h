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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NETLIB_NODELINKCOLLECTION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_NODELINKCOLLECTION_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <vector>
#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "AbstractNodeLinkCollection.h"
#include "NodeLink.h"

using UtilLib::Number;

// Name:   NodeLink
// Author: Marc de Kamps
// Date:   14-07-1999
// Short description: A NodeLinkCollection is basically a complete Network
// description: for every Node in the Network, the related NodeId's are given 
// in a list

namespace NetLib
{

  //! NodeLinkCollection
  class NodeLinkCollection : public AbstractNodeLinkCollection
    {
    public:
	
      //! Read a NodeLinkCollection from an input stream
	  NodeLinkCollection(istream&);

      //! Establish a NodeLinkCollection from a vector of NodeLinks (canonical method) 
      NodeLinkCollection(const vector<NodeLink>&);

      //! Copy constructor
      NodeLinkCollection(const NodeLinkCollection&);

      //! virtual destructor
      virtual ~NodeLinkCollection();

      //! Total number of Nodes, described by the collection
      virtual Number NumberOfNodes() const;

      //! Number of predecessors for a given NodeId
      virtual Number NumberOfPredecessors( NodeId ) const;

      //! pop a NodeLink from the collection, reduces the internal NodeLink vector  
      virtual NodeLink pop();

      //! Give the n-th element of the current NodeLinkCollection vector
      virtual const NodeLink& operator[](size_t) const;

      //! Every kind of NodeLinkCollection has a cloning operator
      virtual NodeLinkCollection* Clone() const;

      //! Nazoeken
      virtual bool IsValid() const;

      //! Current size of the collection
      Number size					() const;

	  //! serialization tag
	  virtual string Tag() const;

	  //! serialization to stream
	  virtual bool ToStream(ostream&) const;

	  //! sreialization from stream
	  virtual bool FromStream(istream&);

    private:

      vector<NodeLink> ParseInputStream(istream&);

      bool             _b_is_valid;
      vector<NodeLink> _vector_of_node_links; // list of NodeLinks

    }; // end of NodeLinkCollection

  //! Streaming
  ostream& operator<<(ostream&, const NodeLinkCollection&);

} // end of NetLib


#endif // include guard

