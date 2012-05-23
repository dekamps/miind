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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_CLAMLIB_SIMULATIONRESULT_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_SIMULATIONRESULT_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4267)
#pragma warning(disable: 4996)
#endif

#include <TFile.h>
#include "../DynamicLib/LocalDefinitions.h"

#include "SimulationResultIterator.h"

namespace ClamLib {

	//! All simulation results are stored in a root file. To isolate clients from
	//! details of the lay-out of the root file, the root file is wrapped into
	//! a SimulationResult object, which allows a traversal of the results using iterators.

	//! The DynamicNetwork is partitioned in DynamicSubNetwork objects.
	//! The SimulationResult offeres iterators to the DynamicSubNetwork parts of the networks
	//! which in turn can be traversed
	class SimulationResult {
	public:
		typedef SimulationResultIterator iterator;

		//! Create a SimulationResult object
		SimulationResult(TFile&);

		//! Traverse DynamicSubNetworks (begin)
		iterator begin(){ return SimulationResult::iterator(_vec_sub.begin());}

		//! Traverse DynamicSubNetworks (end)
		iterator end(){ return SimulationResult::iterator(_vec_sub.end()); }

		DynamicLib::Rate RateForIdByTime(Id, DynamicLib::Time) const;

	private:

		vector<DynamicSubNetwork> InitializeSubNetworkVector() const;

		TFile&						_file_ref;
		vector<DynamicSubNetwork>	_vec_sub;
	};
}
#endif // include
