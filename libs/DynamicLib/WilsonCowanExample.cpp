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

#include "WilsonCowanExample.h"
#include "DynamicNetworkCode.h"
#include "DynamicNetworkImplementationCode.h"
#include "WilsonCowanAlgorithm.h"
#include "WilsonCowanParameter.h"
#include "TestDefinitions.h"


using namespace DynamicLib;

D_DynamicNetwork DynamicLib::WilsonCowanExample()
{
	// Create a WilsonCowan parameter for excitatory Nodes
	WilsonCowanParameter excitatory_parameter
				(
					TAU_EXCITATORY,
					F_MAX,
					F_NOISE
				);

	// And for inhibitory Nodes
	WilsonCowanParameter inhibitory_parameter
				(
					TAU_INHIBITORY,
					F_MAX,
					F_NOISE
				);

	// Create the Algorithms to initialize the Nodes
	WilsonCowanAlgorithm excitatory_algorithm
				(
					excitatory_parameter
				);

	WilsonCowanAlgorithm inhibitory_algorithm
				(
					inhibitory_parameter
				);

	// Creat a constant background rate Node, first create the algorithm :
	D_RateAlgorithm rate
			(
				1
			);

	// A network must be created with some stream,
	// it may be altered during configuration

	D_DynamicNetwork network;

	NodeId id_excitatory = network.AddNode
				(
					excitatory_algorithm,
					EXCITATORY
				);

	NodeId id_inhibitory = network.AddNode
				( 
					inhibitory_algorithm,
					INHIBITORY
				);
				
	NodeId id_rate       = network.AddNode
				(
					rate,
					INHIBITORY
				);

	bool b_weight = true;

	b_weight &=  network.MakeFirstInputOfSecond
			(
				id_excitatory,
				id_excitatory,
				ALPHA
			);

	b_weight &= network.MakeFirstInputOfSecond
			(
				id_inhibitory,
				id_excitatory,
				BETA
			);

	b_weight &= network.MakeFirstInputOfSecond
			(
				id_excitatory,
				id_inhibitory,
				GAMMA
			);

	b_weight &= network.MakeFirstInputOfSecond
			(
				id_inhibitory,
				id_inhibitory,
				DELTA
			);

	b_weight &= network.MakeFirstInputOfSecond
			(
				id_rate,
				id_inhibitory,
				ETA
			);

	b_weight &= network.MakeFirstInputOfSecond
			(
				id_rate,
				id_excitatory,
				ETA
			);

	if (! b_weight)
		throw DynamicLibException(STR_NETWORK_CREATION_FAILED);

	return network;
	// end of example
}
