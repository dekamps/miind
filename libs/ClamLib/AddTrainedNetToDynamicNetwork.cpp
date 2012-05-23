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

#ifdef WIN32
#pragma warning (disable:4244)
#pragma warning (disable:4267)
#pragma warning (disable:4996)
#endif 

#include <list>
#include "../StructnetLib/StructnetLib.h"
#include "AddTrainedNetToDynamicNetwork.h"
#include "BasicDefinitions.h"
#include "ClamLibException.h"
#include "InputCircuitCreator.h"
#include "InverseInputRate.h"
#include "SimpleCircuitCreator.h"
#include "PerceptronCircuitCreator.h"
#include "LocalDefinitions.h"
#include "SemiSigmoid.h"
#include "SpatPosFromPhysPos.h"
#include "RootConversions.h"

using ClamLib::AbstractCircuitCreator;
using ClamLib::AddTNToDN;
using ClamLib::TrainedNet;
using DynamicLib::AbstractAlgorithm;
using DynamicLib::EXCITATORY;
using DynamicLib::D_DynamicNode;
using DynamicLib::D_RateAlgorithm;
using DynamicLib::WilsonCowanAlgorithm;
using UtilLib::Number;

using namespace ClamLib;
using namespace StructnetLib;
using namespace std;

void AddTNToDN::Convert
(
	const TrainedNet&				net,
	const Pattern<Rate>&			input_field,
	const AbstractCircuitCreator&	creator,
	D_DynamicNetwork*				p_dnet,
	RateFunction					p_rate_function,
	const SpatialPosition&			offset_position
)
{
	


	// A new TrainedNet is associated with the DynamicNetwork, create a new WeightList
	// for that

	// TODO: the +1 must be added so that a NodeId corresponds to a vector index. 
	// There NodeIds run from 1 ... N, so the vector must have size N+1. This error
	// was made before so we need a structural solution.
	_vec_weight_list = vector<CircuitInfo>(net._net.NumberOfNodes() + 1);



	_p_dnet				= p_dnet;
	_p_net				= &net;
	_p_rate_function	= p_rate_function;	

	// loop over each Node
	node_iterator iter_begin = net._net.begin(); // iter points at NodeId(1)
	node_iterator iter_end   = net._net.end();

	PhysicalPosition pos;
	SpatialPosition spat_pos;

	_p_creator = boost::shared_ptr<AbstractCircuitCreator>(creator.Clone());

	// First iteration: create all nodes

	for( node_iterator iter = iter_begin; iter != iter_end; iter++)
	{
		pos = net._net.Position(iter->MyNodeId());
		spat_pos = SpatPosFromPhysPos(pos);
		spat_pos += offset_position;

		CircuitInfo info;

		// make a list that associates the network NodeId with the DynamicNetwork NodeId's
		_p_creator->AddNodes(iter->MyNodeId(),&info);

		// associate it with the current WeightList
		_vec_weight_list[iter->MyNodeId()._id_value] = info;	
	}

	AssociateInputFieldWithInputNodes(input_field);

	// second iteration: insert all weights
	InsertDynamicNetWeights();
}

void AddTNToDN::InsertDynamicNetWeights()
{
	// loop over each Node in the ConnectionistNet
	node_iterator iter_begin = _p_net->_net.begin();
	node_iterator iter_end   = _p_net->_net.end();


	// First iteration: loop over all nodes
	for( node_iterator iter = iter_begin; iter != iter_end; iter++)
	{
		// loop over all its inputs
		const_predecessor_iterator iter_begin_daughter = iter->begin();
		const_predecessor_iterator iter_end_daughter   = iter->end();

		for 
		(
			const_predecessor_iterator iter_daughter = iter_begin_daughter; 
			iter_daughter != iter_end_daughter;
			iter_daughter++ 
		)
			_p_creator->AddWeights
			(
				_vec_weight_list[iter->MyNodeId()._id_value],
				_vec_weight_list[iter_daughter->MyNodeId()._id_value],
				iter_daughter.GetWeight()
			);
	}
}


bool AddTNToDN::IsInputNeuron(const PhysicalPosition& pos) const
{
	return (pos._position_z == 0);
}
 
bool AddTNToDN::IsOutputNeuron
(
	const PhysicalPosition& pos
) const
{
	return ( _p_net->_net.Dimensions().size() - 1 == pos._position_z );
}

bool AddTNToDN::IsSymmetricSquashingFunction(const AbstractSquashingFunction& squash) const
{
	double f_max = squash.MaximumActivity();
	double f_min = squash.MinimumActivity();

	if (f_min ==  0.0 && f_max == 1.0)
		return true;
	if (f_min == -1.0 && f_max == 1.0)
		return false;

	throw ClamLibException("Can't handle squashing values");
}

void AddTNToDN::AssociateInputFieldWithInputNodes
(
	const Pattern<Rate>& pat
)
{
	for(Index i = 0; i < pat.Size(); i++)
		// start from NodeId(1)
		if (pat[i] != 0)
			AddRateNode(NodeId(i+1),pat[i]);
		else
			AddZeroRateNode(NodeId(i+1));
	
}

void AddTNToDN::AddZeroRateNode
(
	NodeId i
)
{
	// get info on the node that will receive no input.
	// 'no input' must be actively supplied to push
	// the input neuron to state 0

	CircuitInfo info = _vec_weight_list[i._id_value];

	WilsonCowanParameter par;
	RetrieveParameter(i,&par,&info);
	D_RateAlgorithm rate_alg(InverseInputRate(0.0,par));

	NodeId id_zero_rate = _p_dnet->AddNode(rate_alg,EXCITATORY);

	_p_dnet->MakeFirstInputOfSecond(id_zero_rate,NodeId(info.ExternalInputId()._id_value),1.0);
}

void AddTNToDN::RetrieveParameter
(
	NodeId					id,
	WilsonCowanParameter*	p_wilson,
	CircuitInfo*			p_info
) const
{
	NodeIterator<D_DynamicNode> iter_node = _p_dnet->begin() + id._id_value;
	// Get a clone of the algorithm
	auto_ptr<AbstractAlgorithm<D_DynamicNode::WeightType> > p_alg = iter_node->CloneAlgorithm();

	// which we know is either a WilsonCowan or a SemiSigmoid
	// TODO: this is a strange solution, we need a generic parameter mechanism for all algorithms
	if (dynamic_cast<WilsonCowanAlgorithm*>(p_alg.get()))
		*p_wilson = dynamic_cast<WilsonCowanAlgorithm*>(p_alg.get())->Parameter();
	else
		if (dynamic_cast<SemiSigmoid*>(p_alg.get()))
			*p_wilson = dynamic_cast<SemiSigmoid*>(p_alg.get())->Parameter();
		else
			throw ClamLibException("Couldn't determine algorithm parameter");

}
void AddTNToDN::AddRateNode
(
	NodeId i,
	Rate r
)
{

	// select the relevant input node in the DynamicNetwork to connect to
	CircuitInfo info = _vec_weight_list[i._id_value];

	WilsonCowanParameter par;
	RetrieveParameter(i,&par,&info);


	if (_p_rate_function == 0)
	{
		D_RateAlgorithm rate_alg(InverseInputRate(r,par));
		NodeId id_rate = _p_dnet->AddNode(rate_alg,EXCITATORY);
		_p_dnet->MakeFirstInputOfSecond(id_rate,ToNetNodeId(info.ExternalInputId()),1.0);
	}
	else
	{
		InvertableFunctor func_alg(_p_rate_function, par);
		NodeId id_func = _p_dnet->AddNode(func_alg, EXCITATORY);
		_p_dnet->MakeFirstInputOfSecond(id_func,ToNetNodeId(info.ExternalInputId()),1.0);
	}
}



AddTNToDN::InvertableFunctor::InvertableFunctor
(
	DynamicLib::RateFunction				p_rate_function,
	const DynamicLib::WilsonCowanParameter&	par
 ):
D_AbstractAlgorithm(0),
_par(par),
_p_rate_function(p_rate_function),
_current_time(0.0),
_current_rate(0.0)
{
}

AddTNToDN::InvertableFunctor::~InvertableFunctor()
{
}

bool AddTNToDN::InvertableFunctor::EvolveNodeState
(
	predecessor_iterator iter_begin, 
	predecessor_iterator iter_end, 
	DynamicLib::Time time
)
{
	_current_time = time;

	// as the current time here is set, it makes sense to compute the current rate
	Rate rate = _p_rate_function(time);

	// and now it must be inverted
	_current_rate = InverseInputRate(rate,_par);
	
	return true;
}

Time AddTNToDN::InvertableFunctor::CurrentTime() const
{
	return _current_time;
}

Rate AddTNToDN::InvertableFunctor::CurrentRate() const
{
	return _current_rate;
}

AlgorithmGrid AddTNToDN::InvertableFunctor::Grid() const
{
	return AlgorithmGrid(vector<double>(1,_current_rate));
}

D_AbstractAlgorithm* AddTNToDN::InvertableFunctor::Clone() const
{
	return new InvertableFunctor(*this);
}

NodeState AddTNToDN::InvertableFunctor::State() const
{
	return vector<double>(1,_current_rate);
}

bool AddTNToDN::InvertableFunctor::Configure(const DynamicLib::SimulationRunParameter &)
{
	return true;
}

bool AddTNToDN::InvertableFunctor::Dump(std::ostream &) const
{
	return true;
}

string AddTNToDN::InvertableFunctor::LogString() const
{
	return string("");
}

const std::vector<CircuitInfo>&
	AddTNToDN::CircuitInfoVector() const
{
	return _vec_weight_list;
}