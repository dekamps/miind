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

#ifndef _CODE_LIBS_NETLIB_ABSTRACTARCHITECTURE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_ABSTRACTARCHITECTURE_INCLUDE_GUARD

#include "../UtilLib/UtilLib.h"
//#include "AbstractNodeLinkCollection.h" ---
#include "NodeId.h"


using UtilLib::Number;



namespace NetLib
{

	class AbstractNodeLinkCollection;

	class AbstractArchitecture
    {
    public: 

		//! A given structure, defined by NodeLinkCollection
		AbstractArchitecture
		(
			AbstractNodeLinkCollection*, 
			bool b_threshold = false
		);

		virtual ~AbstractArchitecture() = 0;

		   
		//! copy constructor
		AbstractArchitecture(const AbstractArchitecture&);

		//! copy operator
		AbstractArchitecture& operator=(const AbstractArchitecture&);

		//! Total number of nodes descibed by the Architecture
		Number NumberOfNodes() const;

		//! Number of input nodes may be defined differently for derived classes	
		virtual UtilLib::Number NumberOfInputNodes() const = 0;

		//! Number of output nodes may be defined differently for derived classes
		virtual UtilLib::Number NumberOfOutputNodes() const = 0;

		//! Number of connections
		virtual UtilLib::Number NumberOfConnections() const;

		//! Give acces to the NodeLinkCollection
		AbstractNodeLinkCollection* Collection();

		//! const version
		const AbstractNodeLinkCollection* Collection() const;

		//! Test if the associated NodeLinkCollection was used
		bool IsCollectionEmpty() const;

		//! Test if a threshold was specified
		bool HaveAThreshold() const;

    protected:

		Number ConnectionCount
	    (
			NodeId id, 
			Number nr_nodes
	    ) const ;

    private:

		bool						_b_threshold;
		Number						_number_of_nodes;
		AbstractNodeLinkCollection*	_p_collection;

		// Order is important: NodeLinkCollection must be initialized
		// before _number_of_connections can be calculated

		Number						_number_of_connections;

    }; // end of AbstractArchitecture

} // end of NetLib

#endif // include guard
