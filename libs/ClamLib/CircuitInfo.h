// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_CLAMLIB_CIRCUITINFO_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_CIRCUITINFO_INCLUDE_GUARD

#include <TNamed.h>
#include "Id.h"
#include "CircuitDescription.h"


namespace ClamLib {

	//! CircuitInfo:
	struct CircuitInfo : public TNamed {
	public:

		ClassDef(CircuitInfo,1);

		CircuitInfo();

		Id ExternalInputId() const;

		Id& operator[](int);

		const Id& operator[](int) const;

		const Id& IdOriginal() const;

		//! Associate a given circuit with an Artificial Neural Network Id. 
		//! Called by the AbstractCircuitCreator whose name will be entered in the CircuitInfo, Reserve reserves
		//! precisely the number of nodes associated with the Circuit.
		void Reserve(const char * Name, int, Id );

		int NumberOfNodes() const;

		void SetExternal(ClamLib::Id);

	private:

		Id					_id_ann;
		Id					_id_external;
		std::vector<Id>		_vec_id;
	};

}

#endif // include guard
