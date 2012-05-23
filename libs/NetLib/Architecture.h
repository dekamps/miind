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
#ifndef _CODE_LIBS_NETLIB_ARCHITECTURE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_ARCHITECTURE_INCLUDE_GUARD

// Purpose: Every network implementation starts from some Architecture
// Created: 14-07-1999
// Author:  Marc de Kamps

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <vector>
#include <utility>
#include <cassert>
#include "../UtilLib/UtilLib.h"
#include "AbstractArchitecture.h"
#include "BasicDefinitions.h"
#include "NodeLinkCollection.h"

using UtilLib::Number;
using UtilLib::Index;

namespace NetLib
{

	//! Architecture

	//! Mathematical shorthand for network structures
	class Architecture : public AbstractArchitecture
	{
	public:


		//! Default: fully connected
		Architecture
			(
				Number, 
				bool b_threshold = false
			);

		//! 
		Architecture
			(
				NodeLinkCollection*,
				bool b_threshold = false
			);
		//! derived classes: virtual destructor
		virtual ~Architecture();			
			   
		//! copy constructor
		Architecture(const Architecture&);

		//! copy operator
		Architecture& operator=(const Architecture&);

		//! Number of input nodes may be defined differently for derived classes
		virtual Number NumberOfInputNodes() const;

		//! Number of output nodes may be defined differently for derived classes
		virtual Number NumberOfOutputNodes() const;

		//! Number of connections
		virtual Number NumberOfConnections() const;
	

	private:

		Number _number_of_connections;

		AbstractNodeLinkCollection* _p_collection;

	}; // end of Architecture

} // end of NetLib

#endif // include guard
