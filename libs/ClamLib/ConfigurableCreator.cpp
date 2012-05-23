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

#include <boost/foreach.hpp>
#include "Id.h"
#include "ConfigurableCreator.h"
#include "RootConversions.h"

using namespace ClamLib;
using namespace DynamicLib;

ConfigurableCreator::ConfigurableCreator
(
	const D_AbstractAlgorithm*	p_alg_exc,
	const D_AbstractAlgorithm*	p_alg_inh,
	D_DynamicNetwork*			p_net,
	const CircuitDescription&	desc
):
  AbstractCircuitCreator(p_net),
_p_alg_exc(p_alg_exc),
_p_alg_inh(p_alg_inh),
_p_net(p_net),
_desc(desc)
{
}

ConfigurableCreator::~ConfigurableCreator()
{
}

void ConfigurableCreator::AddNodes(NodeId idc, CircuitInfo* p_info)
{
	p_info->Reserve(_desc.GetName(),_desc.size(),Id(idc._id_value));
	const vector<CircuitNodeRole>& vec_role = _desc.RoleVec();
	int index = 0;
	BOOST_FOREACH(const CircuitNodeRole& role,vec_role){
		// here the nodes that are in the circuits are inserted in the network
		(*p_info)[index++] = ToClamNodeId(_p_net->AddNode(*_p_alg_exc,static_cast<NodeType>(role.Type())));
		// keep track of the relation between id in the role and Id
	}
	
	index = 0;
	// Now that all the nodes in the network have been created, we can  insert the Circuit connections
	BOOST_FOREACH(const CircuitNodeRole& roleconnect, vec_role)
	{
		const vector<IndexWeight>& vec_weight = roleconnect.IncomingVec();
		BOOST_FOREACH(const IndexWeight& iw, vec_weight){
			_p_net->MakeFirstInputOfSecond
			(
				ToNetNodeId((*p_info)[iw._index]),
				ToNetNodeId((*p_info)[index]),
				iw._weight
			);
		}
		index++;
	}
	// finally set the external node
	p_info->SetExternal((*p_info)[_desc.IndexExternal()]);

}

ConfigurableCreator* ConfigurableCreator::Clone() const
{
	return new ConfigurableCreator(*this);
}

UInt_t ConfigurableCreator::NumberOfNodes() const
{
	return _desc.size();
}

string ConfigurableCreator::Name() const
{
	return _desc.GetName();
}

void ConfigurableCreator::AddWeights
(
	const CircuitInfo& info_out, 
	const CircuitInfo& info_in, 
	ClamLib::Efficacy  weight
)
{
	const vector<InputOutputPair>& vec_io = _desc.IOVec();

	BOOST_FOREACH(const InputOutputPair& iopair, vec_io){
		_p_net->MakeFirstInputOfSecond
			(
				ToNetNodeId(info_in[iopair._id_in]),
				ToNetNodeId(info_out[iopair._id_out]),
				weight
			);
	}
}

CircuitDescription ConfigurableCreator::Description() const
{
	return _desc;
}