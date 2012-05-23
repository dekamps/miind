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

#include "PerceptronCircuitCreator.h"
#include "CreatorStockObjects.h"
#include "ClamLibException.h"
#include "LocalDefinitions.h"
#include "RootConversions.h"


using namespace ClamLib;

using DynamicLib::D_AbstractAlgorithm;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::EXCITATORY;
using DynamicLib::INHIBITORY;


PerceptronCircuitCreator::PerceptronCircuitCreator
(
	const D_AbstractAlgorithm*	p_exc_alg,
	const D_AbstractAlgorithm*	p_inh_alg,
	D_DynamicNetwork*			p_dnet,
	const SpatialPosition&		pos
):
AbstractCircuitCreator(p_dnet),
_p_exc_alg(p_exc_alg),
_p_inh_alg(p_inh_alg),
_p_dnet(p_dnet),
_pos(pos)
{
	// check if the inhibitory algorithm is actually defined
	if (! _p_inh_alg)
		throw ClamLibException("Inhibitory algorithm required");
}

PerceptronCircuitCreator::PerceptronCircuitCreator():
AbstractCircuitCreator(0),
_p_exc_alg(0),
_p_inh_alg(0),
_p_dnet(0),
_pos(NO_OFFSET)
{
}

PerceptronCircuitCreator* PerceptronCircuitCreator::Clone() const
{
	return new PerceptronCircuitCreator(*this);
}

void PerceptronCircuitCreator::AddNodes
(
	NodeId id,
	CircuitInfo* p_info
)
{
	p_info->Reserve(this->Name().c_str(),this->NumberOfNodes(),Id(id._id_value));

	(*p_info)[P_OUT] = ToClamNodeId(_p_dnet->AddNode(*_p_exc_alg,EXCITATORY));
	(*p_info)[N_OUT] = ToClamNodeId(_p_dnet->AddNode(*_p_exc_alg,EXCITATORY));
	(*p_info)[EP]    = ToClamNodeId(_p_dnet->AddNode(*_p_exc_alg,EXCITATORY));
	(*p_info)[EN]    = ToClamNodeId(_p_dnet->AddNode(*_p_exc_alg,EXCITATORY));
	(*p_info)[IN]    = ToClamNodeId(_p_dnet->AddNode(*_p_inh_alg,INHIBITORY));
	(*p_info)[IP]    = ToClamNodeId(_p_dnet->AddNode(*_p_inh_alg,INHIBITORY));

	// id to which external inputs must couple 
	p_info->SetExternal((*p_info)[EP]);

	this->AssociateNodesWithPositions(*p_info);

	_p_dnet->MakeFirstInputOfSecond(ToNetNodeId((*p_info)[EP]),ToNetNodeId((*p_info)[P_OUT]),CIRCUIT_WEIGHT);
	_p_dnet->MakeFirstInputOfSecond(ToNetNodeId((*p_info)[IP]),ToNetNodeId((*p_info)[N_OUT]),-CIRCUIT_WEIGHT);
	_p_dnet->MakeFirstInputOfSecond(ToNetNodeId((*p_info)[EN]),ToNetNodeId((*p_info)[N_OUT]),CIRCUIT_WEIGHT);
	_p_dnet->MakeFirstInputOfSecond(ToNetNodeId((*p_info)[IN]),ToNetNodeId((*p_info)[P_OUT]),-CIRCUIT_WEIGHT);
}

void PerceptronCircuitCreator::AssociateNodesWithPositions(const CircuitInfo& info)
{


	SpatialPosition pos_p_out = _pos;
	pos_p_out._x += PC_EP_X_OFFSET;
	pos_p_out._z += PC_EP_Z_OFFSET;
	_p_dnet->AssociateNodePosition(ToNetNodeId(info[P_OUT]), pos_p_out);

	SpatialPosition pos_n_out = _pos;
	pos_n_out._x -= PC_EP_X_OFFSET;
	pos_n_out._z += PC_EP_Z_OFFSET;
	_p_dnet->AssociateNodePosition(ToNetNodeId(info[N_OUT]), pos_n_out);

	SpatialPosition pos_ep = _pos;
	pos_ep._x += 1.5F*PC_EP_X_OFFSET;
	_p_dnet->AssociateNodePosition(ToNetNodeId(info[EP]), pos_ep);

	SpatialPosition pos_en = _pos;
	pos_en._x -= 1.5F*PC_EP_X_OFFSET;
	_p_dnet->AssociateNodePosition(ToNetNodeId(info[EN]), pos_en);

	SpatialPosition pos_ip = _pos;
	pos_ip._x += 0.5F*PC_EP_X_OFFSET;
	_p_dnet->AssociateNodePosition(ToNetNodeId(info[IP]), pos_ip);

	SpatialPosition pos_in = _pos;
	pos_in._x -= 0.5F*PC_EP_X_OFFSET;
	_p_dnet->AssociateNodePosition(ToNetNodeId(info[IN]), pos_in);
}


Number PerceptronCircuitCreator::NumberOfNodes() const
{
	return _nr_perceptron_populations;
}

string PerceptronCircuitCreator::Name() const
{
	return _name;
}

const Number PerceptronCircuitCreator::_nr_perceptron_populations = 6;

const string PerceptronCircuitCreator::_name("PerceptronCircuitCreator");

void PerceptronCircuitCreator::AddWeights
(
	const CircuitInfo& info_out, 
	const CircuitInfo& info_in, 
	ClamLib::Efficacy weight
)
{
	if (string(info_out.GetName()) == string(info_in.GetName()) && string(info_out.GetName()) == this->Name() )
	{
		// if weight is positive connect P_OUT to E_P, I_P; N_OUT to E_N, I_N
		// if weight is negative connect P_OUT to E_N, I_N; N_OUT to E_P, I_P
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
	throw ClamLibException("Didn't know how to connect nodes");
}

CircuitDescription PerceptronCircuitCreator::Description() const
{
	return CreatePerceptronDescription();
}