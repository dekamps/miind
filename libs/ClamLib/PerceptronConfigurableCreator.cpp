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
#pragma warning(disable: 4267 4996)
#endif

#include "PerceptronConfigurableCreator.h"

using namespace ClamLib;

PerceptronConfigurableCreator::PerceptronConfigurableCreator
(
	const D_AbstractAlgorithm*	p_exc_alg,
	const D_AbstractAlgorithm*	p_inh_alg,
	D_DynamicNetwork*			p_dnet,
	const CircuitDescription&	desc
):
ConfigurableCreator(p_exc_alg,p_inh_alg,p_dnet,desc),
_p_exc_alg(p_exc_alg),
_p_inh_alg(p_inh_alg),
_p_dnet(p_dnet),
_desc(desc)
{
}

PerceptronConfigurableCreator::~PerceptronConfigurableCreator()
{
}

PerceptronConfigurableCreator*
	PerceptronConfigurableCreator::Clone() const
{
	return new PerceptronConfigurableCreator(*this);
}

void PerceptronConfigurableCreator::AddWeights
(
	const CircuitInfo& info_out, 
	const CircuitInfo& info_in, 
	ClamLib::Efficacy  weight
)
{
	Index P_OUT = _desc.IndexInCircuitByName("P_OUT");
	Index N_OUT = _desc.IndexInCircuitByName("N_OUT");
	Index EP    = _desc.IndexInCircuitByName("e_p");
	Index EN    = _desc.IndexInCircuitByName("e_n");
	Index IP    = _desc.IndexInCircuitByName("i_p");
	Index IN    = _desc.IndexInCircuitByName("i_n");

	if (weight >= 0)
	{
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[P_OUT]._id_value),NodeId(info_out[EP]._id_value),weight);
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[P_OUT]._id_value),NodeId(info_out[IP]._id_value),weight);
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[N_OUT]._id_value),NodeId(info_out[EN]._id_value),weight);
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[N_OUT]._id_value),NodeId(info_out[IN]._id_value),weight);
	}
	else
	{
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[P_OUT]._id_value),NodeId(info_out[EN]._id_value),-weight);
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[P_OUT]._id_value),NodeId(info_out[IN]._id_value),-weight);
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[N_OUT]._id_value),NodeId(info_out[EP]._id_value),-weight);
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[N_OUT]._id_value),NodeId(info_out[IP]._id_value),-weight);
	}
	return;
}