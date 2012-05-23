#include "DynamicLib/DynamicLib.h"
#include "WalrasianTraderAlgorithm.h"
#include <vector>
#include <iostream>

using namespace DynamicLib;
using namespace SparseImplementationLib;
using std::cout;
using std::endl;



WalrasianTraderAlgorithm::WalrasianTraderAlgorithm
(	
	const vector<double>& state,
	string name
):
AbstractAlgorithm<double>(1)
{
	_state = state;
	_time = 0;
	_name = name;
}

WalrasianTraderAlgorithm::~WalrasianTraderAlgorithm(void)
{
}

//! evolve node state
bool WalrasianTraderAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{
	_time = time;

	predecessor_iterator cii;
	
	D_AbstractSparseNode* p_node = &(*begin_iter); 
	D_DynamicNode* p_dyn = dynamic_cast<D_DynamicNode*>(p_node);

	

	if (p_dyn)
	{
		auto_ptr<AbstractAlgorithm<double> > algorithm = p_dyn->CloneAlgorithm();
		NodeState vec = algorithm->State();
		
		_state[1] = vec[1];
		_state[2] = vec[2];

		//cout << "mine: " << _state[0] << " " << _state[1] << " " << _state[2] << endl;
		//cout << "other: " << vec[0] << " " << vec[1] << " " << vec[2] << endl;
		//cout << "other: " << p_dyn->State() << endl;
	}

	return true;		
}

//! clone
WalrasianTraderAlgorithm* WalrasianTraderAlgorithm::Clone() const
{
	return new WalrasianTraderAlgorithm(*this);
}

//! report initial state
AlgorithmGrid WalrasianTraderAlgorithm::Grid() const
{
	return _state;
}

//! report node state
NodeState WalrasianTraderAlgorithm::State() const
{
	//cout << "inside WalrasianTraderAlgorithm::State()" << endl;
	NodeState ns(_state);
	//cout << &ns << " " << ns[0] << " " << ns[1] << " " << ns[2] << endl;
	return ns;
}

//!
string WalrasianTraderAlgorithm::LogString() const 
{
	return string("hi");
}

//! streaming to output stream
bool WalrasianTraderAlgorithm::ToStream(ostream&) const
{
	return true;
}

//! streaming from input stream
bool WalrasianTraderAlgorithm::FromStream(istream&)
{
	return true;
}

//! stream Tag
string WalrasianTraderAlgorithm::Tag() const
{
	return _name;	
}

bool WalrasianTraderAlgorithm::Dump(ostream&) const
{
	return true;
}

bool WalrasianTraderAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	return true;
}

Time WalrasianTraderAlgorithm::CurrentTime() const
{
	return _time;
}

Rate WalrasianTraderAlgorithm::CurrentRate() const
{
	return 1;
}
