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
#pragma warning(disable:4267)
#pragma warning(disable:4996)
#endif

#include "BasicDefinitions.h"
#include "CreatorStockObjects.h"
#include "SimpleCircuitCreator.h"
#include "RootConversions.h"

using namespace ClamLib;
using DynamicLib::D_AbstractAlgorithm;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::EXCITATORY;
using DynamicLib::SpatialPosition;


SimpleCircuitCreator::SimpleCircuitCreator():
AbstractCircuitCreator(0),
_p_exc_alg(0),
_p_dnet(0),
_pos(NO_OFFSET)
{
}

SimpleCircuitCreator* SimpleCircuitCreator::Clone() const
{
	return new SimpleCircuitCreator(*this);
}

SimpleCircuitCreator::SimpleCircuitCreator
(
	const D_AbstractAlgorithm*	p_exc_alg,
	const D_AbstractAlgorithm*	p_inh_alg,
	D_DynamicNetwork*			p_dnet,
	const SpatialPosition&		pos
):
AbstractCircuitCreator(p_dnet),
_p_exc_alg(p_exc_alg),
_p_dnet(p_dnet),
_pos(pos)
{
}


void SimpleCircuitCreator::AddNodes
(
	NodeId			idc,	
	CircuitInfo*	p_info
)
{

	Id id_node = ToClamNodeId(_p_dnet->AddNode(*_p_exc_alg,EXCITATORY));
	p_info->Reserve("SimpleCircuitCreator",this->NumberOfNodes(),Id(idc._id_value));
	(*p_info)[0] = id_node;
	p_info->SetExternal(id_node);
	_p_dnet->AssociateNodePosition(ToNetNodeId(id_node),_pos);
}

Number SimpleCircuitCreator::NumberOfNodes() const
{
	return _nr_simple_populations;
}

string SimpleCircuitCreator::Name() const
{
	return _name;
}

void SimpleCircuitCreator::AddWeights
(
	const CircuitInfo& info_out, 
	const CircuitInfo& info_in, 
	ClamLib::Efficacy weight
)
{

	if (info_out.NumberOfNodes() == 1 && info_in.NumberOfNodes() == 1)
	{
		// just connect the first ones in the list
		_p_dnet->MakeFirstInputOfSecond(NodeId(info_in[0]._id_value),NodeId(info_out[0]._id_value),weight);
		return;
	}
}

const Number SimpleCircuitCreator::_nr_simple_populations = 1;
const string SimpleCircuitCreator::_name("SimpleCircuitCreator");

CircuitDescription SimpleCircuitCreator::Description() const
{
	return CreateSingleDescription();
}