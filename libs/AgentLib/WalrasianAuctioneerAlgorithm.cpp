#include "DynamicLib/DynamicLib.h"
#include "WalrasianAuctioneerAlgorithm.h"
#include "WalrasianTraderAlgorithm.h"
#include <vector>
#include <iostream>
#include <map>

using namespace DynamicLib;
using namespace SparseImplementationLib;
using std::cout;
using std::endl;



WalrasianAuctioneerAlgorithm::WalrasianAuctioneerAlgorithm
(	
	const vector<double>& state,
	string tag
):
AbstractAlgorithm<double>(1)
{
	_state = state;
	_time = 0;
	_tag = tag;
}

WalrasianAuctioneerAlgorithm::~WalrasianAuctioneerAlgorithm(void)
{
}

//! evolve node state
bool WalrasianAuctioneerAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{
	_time = time;

	cout << time << endl;

	predecessor_iterator cii;
	
	
	cout << "mine: " << _state[0] << " " << _state[1] << " " << _state[2] << endl;
	
	// data gathering structures
	int trader_counter = 0;
	map <string, double*> data;

	// retrieving data from traders
	D_AbstractSparseNode* p_node;
	D_DynamicNode* p_dyn;
	double* demand_supply;
	for(cii=begin_iter; cii!=end_iter; cii++, trader_counter++)
	{
		p_node = &(*cii); 
		p_dyn = dynamic_cast<D_DynamicNode*>(p_node);	

		if (p_dyn)
		{
			auto_ptr<AbstractAlgorithm<double> > algorithm = p_dyn->CloneAlgorithm();
			NodeState vec = algorithm->State();
			//cout << algorithm->Tag() << ": " << vec[0] << " " << vec[1] << " " << vec[2] << endl;

			demand_supply = (double*) calloc (2, sizeof(double));
			demand_supply[0] = vec[1];
			demand_supply[1] = vec[2];

			data.insert(pair<string, double*>(algorithm->Tag(), demand_supply));
		}
	}

	double total_demand = 0, total_supply = 0;

	for(map<string, double*>::const_iterator it = data.begin(); it != data.end(); ++it)
    {
        cout << "Trader: " << it->first;
		cout << " Demand: " << it->second[0] << " Supply: " << it->second[1] << endl;

		total_demand += it->second[0];
		total_supply += it->second[1];
    }


	
	_state[1] = (total_demand/trader_counter)+2;
	_state[2] = (total_supply/trader_counter)+2;

	return true;		
}

//! clone
WalrasianAuctioneerAlgorithm* WalrasianAuctioneerAlgorithm::Clone() const
{
	return new WalrasianAuctioneerAlgorithm(*this);
}

//! report initial state
AlgorithmGrid WalrasianAuctioneerAlgorithm::Grid() const
{
	return _state;
}

//! report node state
NodeState WalrasianAuctioneerAlgorithm::State() const
{
	//cout << "inside WalrasianAuctioneerAlgorithm::State()" << endl;
	NodeState ns(_state);
	//cout << &ns << " " << ns[0] << " " << ns[1] << " " << ns[2] << endl;
	return ns;
}

//!
string WalrasianAuctioneerAlgorithm::LogString() const 
{
	return string("hi");
}

//! streaming to output stream
bool WalrasianAuctioneerAlgorithm::ToStream(ostream&) const
{
	return true;
}

//! streaming from input stream
bool WalrasianAuctioneerAlgorithm::FromStream(istream&)
{
	return true;
}

//! stream Tag
string WalrasianAuctioneerAlgorithm::Tag() const
{
	return _tag;	
}

bool WalrasianAuctioneerAlgorithm::Dump(ostream&) const
{
	return true;
}

bool WalrasianAuctioneerAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	return true;
}

Time WalrasianAuctioneerAlgorithm::CurrentTime() const
{
	return _time;
}

Rate WalrasianAuctioneerAlgorithm::CurrentRate() const
{
	return _state[1];
}
