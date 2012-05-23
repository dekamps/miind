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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_DYNAMICLIB_AFFERENT_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_AFFERENT_INCLUDE_GUARD

#include "DynamicNode.h"
#include "LocalDefinitions.h"

using SparseImplementationLib::AbstractSparseNode;

namespace DynamicLib
{

	//! An auxilliary class used for test purposes. It can provide input to an AbstractAlgorithm without having been derived from AbstractAlgorithm or encapsulated in a Node. One
	//! should not use this class in DynamicNetwork. External input into a network is provided by RateFunctor, which is a DynamicNode in its own right.

	//! This class provides and artificial scalar product for testing AbstractAlgorithm. The connection vector simulated by this class has a single connection with weight value 1. Internally,
	//! it stores a DynamicNode with activation set to the input rate, so that the value of this rate is read by the Algorithm. In this way the AbstractAlgorithm is guaranteed to 'see'
	//! scalar product of value rate. Clients are unlikely to ever need this class, except, perhaps, for testing their own AbstractAlgorithm derived classes.
	template <class WeightValue>
	class Afferent
	{
	public:

		typedef DynamicNode<WeightValue> node;
		typedef typename AbstractSparseNode<double,WeightValue>::predecessor_iterator predecessor_iterator;
		typedef pair< AbstractSparseNode<double,WeightValue>*, WeightValue > connection;
		
		//! Only an Afferent with a fixed Rate can be produced.
		Afferent(Rate rate); 
	
		//! begin iterator to a connection vector is provided for use by an AbstractAlgorithm
		predecessor_iterator begin();

		//! end iterator to a connection vector is provided for use by an AbstractAlgorithm
		predecessor_iterator end();

	private:

		node   _the_node;
		double _f_rate;

		vector<connection> _vector_of_connections;

	}; // end of Afferent

	typedef Afferent<double> D_Afferent;

} // end of DynamicLib

#endif // include guard

