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
#pragma warning(disable: 4267 4996 4244)
#endif

#include "../DynamicLib/DynamicLib.h"
#include "CreatorStockObjects.h"

using namespace ClamLib;
using namespace DynamicLib;

CircuitDescription ClamLib::CreateSingleDescription()
{
	// emulate the functionality of the SimpleCircuitCreator
	CircuitNodeRole role("the_node",DynamicLib::EXCITATORY,0.0,0.0,0.0,0.0, true);

	CircuitDescription desc(1);
	desc.AddExternal("the_node"); // now necessary, add before finalizing the node
	Index id_single = desc.push_back(role);

	InputOutputPair pair;
	pair._id_in  = id_single;
	pair._id_out = id_single;
	desc.push_back_io(pair);
	desc.SetName("simple");

	return desc;
}

CircuitDescription ClamLib::CreatePerceptronDescription()
{
	CircuitDescription desc(6);

	IndexWeight iw;

	static const bool OUTPUT = true;
	static const bool POSITIVE = true;
	static const bool NEGATIVE = false;
	CircuitNodeRole p_out("P_OUT",DynamicLib::EXCITATORY, 1.0,0.0,1.0,0.0, OUTPUT, POSITIVE);
	CircuitNodeRole n_out("N_OUT",DynamicLib::EXCITATORY,-1.0,0.0,1.0,0.0, OUTPUT, NEGATIVE);
	CircuitNodeRole e_p  ("e_p",  DynamicLib::EXCITATORY, 1.5,0.0,0.0,0.0);
	CircuitNodeRole e_n  ("e_n",  DynamicLib::EXCITATORY,-1.5,0.0,0.0,0.0);
	CircuitNodeRole i_n  ("i_n",  DynamicLib::INHIBITORY, 1.5,0.0,0.0,0.0);
	CircuitNodeRole i_p  ("i_p",  DynamicLib::INHIBITORY, 0.5,0.0,0.0,0.0);

	double circuit_weight = 2.0;
	p_out.AddIncoming(IndexWeight("e_p",circuit_weight));
	n_out.AddIncoming(IndexWeight("i_p",-circuit_weight));
	n_out.AddIncoming(IndexWeight("e_n",circuit_weight));
	p_out.AddIncoming(IndexWeight("i_n",-circuit_weight));

	desc.AddExternal("e_p");

	desc.push_back(p_out);
	desc.push_back(n_out);
	desc.push_back(e_p);
	desc.push_back(e_n);
	desc.push_back(i_n);
	desc.push_back(i_p);

	return desc;
}
