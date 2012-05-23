// Copyright (c) 2005 - 2010 Dave Harrison
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


#ifndef _CODE_LIBS_CLAMLIB_CircuitNodeRole_H_
#define _CODE_LIBS_CLAMLIB_CircuitNodeRole_H_

#include <string>
#include <vector>
#include <TNamed.h>
#include <TString.h>
#include "../DynamicLib/SpatialPosition.h"
#include "IndexWeight.h"

using std::string; 
using std::vector;


//!Instances of this class represent the CircuitNodeRole a Node plays within a circuit
//!
//! Each concrete CircuitCreator must create instances of each CircuitNodeRole that are needed by a circuit
//! Each CircuitNodeRole includes a SpatialPosition of that Node within the circuit (not globally),
//! and vectors of incoming and outgoing CircuitNodeRoles that are connected.

namespace ClamLib {

	class CircuitDescription;

	
	class CircuitNodeRole : public TNamed {
	public:

		ClassDef(CircuitNodeRole,1)

		CircuitNodeRole();

		/**
		 * Constructor
		 * \param isOutput true if node is an output node (default = false)
		 * \param isPositive true if node's output is interpreted as positive (default = true)
		 */
		CircuitNodeRole
		(
			const string& name,		//! Name of node within the circuit
			UInt_t,					//! Identifier that labels the node within the circuit
			Float_t,				//! Spatial offset within the circuit: x
			Float_t,				
			Float_t,
			Float_t = 0.0,				//! Spatial offset as a feature
			bool isOutput = false,
			bool isPositive = true
		);

		CircuitNodeRole
		(
			const CircuitNodeRole& role
		): 
		TNamed(role),
		_type(role._type),
		_x(role._x),
		_y(role._y),
		_z(role._z),
		_f(role._f),
		_isOutput(role._isOutput),
		_isPositive(role._isPositive),
		_incoming(role._incoming) 
		{}

		UInt_t size() const;

		DynamicLib::SpatialPosition Position() const;

		// ROOT serialization can't deal with this one:
		//	IndexWeight& operator[](UInt_t);

		const IndexWeight& operator[](UInt_t) const;

		const std::vector<ClamLib::IndexWeight>& 
			IncomingVec() const;

		UInt_t Type() const;

		bool  AddIncoming(IndexWeight);

		//! if node has output role
		//! \return true if node is an output node
		bool isOutput() const;

		//! if node's value is interpreted as a positive value
		//! \return true if node's output is interpreted as a positive value
		bool isPositive() const;

		friend class CircuitDescription;

	private:

		UInt_t						_type; // EXCITATORY or INHIBITORY
		Float_t						_x;
		Float_t						_y;
		Float_t						_z;
		Float_t						_f;	
		bool						_isOutput; // If this node is an output node
		bool						_isPositive; // If node output is considered a positive value

		std::vector<ClamLib::IndexWeight>	_incoming;

	};

	bool operator==(const CircuitNodeRole&, const char*);

} // end of ClamLib

#endif /* CircuitNodeRole_H_ */
