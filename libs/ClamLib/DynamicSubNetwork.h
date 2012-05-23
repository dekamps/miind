
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
#ifdef WIN32
#pragma warning(disable: 4267)
#pragma warning(disable: 4996)
#endif

#ifndef _CODE_LIBS_CLAMLIB_DYNAMICSUBNETWORK_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_DYNAMICSUBNETWORK_INCLUDE_GUARD

#include <string>
#include "ForwardSubNetworkIterator.h"
#include "ReverseSubNetworkIterator.h"
#include "SimulationInfoBlock.h"

using std::string;

namespace ClamLib {

	//! DynamicSubNetwork offers methods to traverse parts of the DynamicNetwork
	//! These parts usually are created by automated methods and have a clear spatial
	//! organization, which is described by a SimulationInfoBlock. 
	
	//! For each SimulationInfoBlock a corresponding DynamicSubNetwork can be generated,
	//! which is able to generate several iterators that enable a systematic traversal
	//! of the corresponding part of the DynamicNetwork. This is important in visualization:
	//! usually a DynamicSubNetwork must be rendered as a whole and the iterators give
	//! a systematic access to the NodeId which labels the TGraphs. A second important
	//! application is to systematically create circuits that connect to parts of the network.
	//! A derefence of a ForwardSubNetworkIterator yields a reference to a CircuitInfo, i.e. these
	//! iterators give access to the circuits that were produced in each layer.
	class DynamicSubNetwork {
	public:
		typedef ForwardSubNetworkIterator const_iterator;
		typedef ReverseSubNetworkIterator const_rziterator;

		//! DynamicSubNetwork needs to be strored in vectors
		DynamicSubNetwork();

		//! Create a DynamicSubNetwork from its SimulationInfoBlock
		DynamicSubNetwork(const SimulationInfoBlock&);

		//! Give a systematic traversal of the network through iterators
		const_iterator begin() const;

		//! Give a systematic traversal of the network through iterators (end)
		const_iterator end() const;

		//! Iterate through one layer in the network
		const_iterator begin(Layer) const;

		//! Iterate through one layer in the network (end)
		const_iterator end(Layer) const;

		//! Traverse a generated feedback network from output to input, in such a way that the
		//! PhysicalPosition of the corresponding feedforward network is matched
		const_rziterator rzbegin() const;
		//!
		const_rziterator rzend() const;

		const_rziterator rzbegin(Layer) const;

		const_rziterator rzend(Layer) const;

		//! Name of corresponding SimulationInfoBlock
		string GetName() const;

		void Write() const;

		//! Get the circuit description object from the SimulationInfoBlock
		const CircuitDescription& GetCircuitDescription() const;

		friend class SimulationInfoBlock;

	private:

		SimulationInfoBlock _block;

		static const string EXCEP;
	};

} // end of ClamLib

#endif // include guard
