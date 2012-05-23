#include "MarketMakerAlgorithm.h"
#include "RandSimulator.h"


MarketMakerAlgorithm::MarketMakerAlgorithm
(	
	const vector<double>& state,
	int mode,
	double equilibrium_price,
	bool equilibrium_stable
):
AbstractAlgorithm<double>(1)
{
	_state = state;
	_time = 0;
	_mode = mode;
	_state[1] = equilibrium_price;
	_equilibrium_stable = equilibrium_stable;
	//srand ( time(NULL) );
}

MarketMakerAlgorithm::~MarketMakerAlgorithm(void)
{
}

//! evolve node state
bool MarketMakerAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{
	_time = time;

	//Sleep(5);

	// retrieving data from traders
	D_AbstractSparseNode* p_node;
	D_DynamicNode* p_dyn;
	double total_demand=_state[0];

	//total_demand = 0;

	predecessor_iterator cii;
	int trader_count = 0;
	for(cii=begin_iter; cii!=end_iter; cii++, trader_count++)
	{
		p_node = &(*cii); 
		p_dyn = dynamic_cast<D_DynamicNode*>(p_node);	

		if (p_dyn)
		{
			auto_ptr<AbstractAlgorithm<double> > algorithm = p_dyn->CloneAlgorithm();
			total_demand += algorithm->State()[0];
			if(algorithm->State()[1]!=-1)
			{
				_psi = algorithm->State()[1];
				_cd = algorithm->State()[0];
			}
			else
				_fd = algorithm->State()[0];
		}
	}

	//cout << " mm " <<_state[0] << endl;

	_state[0] = total_demand;

	//cout << "M:" << _state[0] << endl;;

	if(trader_count == 0)
		return true;

	switch(_mode)
	{
		case RANDOM:
			_state[2] = AgentLib::RandSimulator() % trader_count + 1;
			break;

		case SEQUENTIAL:
			if(_state[2] == trader_count)
				_state[2] = 1;
			else
				_state[2]++;
			break;

		case SYNCHRONOUS:
			_state[2] = -1;
			break;
	}

	if(_equilibrium_stable == false)
	{
		if(AgentLib::RandSimulator()%2 == 1)
			_state[1] += 0.1;
		else
			_state[1] -= 0.1;
		cout << "eq\t"<< _state[1] << endl;
	}

	//cout << _state[0] << endl;

	return true;		
}

//! clone
MarketMakerAlgorithm* MarketMakerAlgorithm::Clone() const
{
	return new MarketMakerAlgorithm(*this);
}

//! report initial state
AlgorithmGrid MarketMakerAlgorithm::Grid() const
{
	return _state;
}

//! report node state
NodeState MarketMakerAlgorithm::State() const
{
	NodeState ns(_state);
	return ns;
}

//!
string MarketMakerAlgorithm::LogString() const 
{
	char s[1000];
	sprintf ( s, "m%f %f %f %f", _state[1], _psi, _fd, _cd);
	return string(s);
}

//! streaming to output stream
bool MarketMakerAlgorithm::ToStream(ostream&) const
{
	return true;
}

//! streaming from input stream
bool MarketMakerAlgorithm::FromStream(istream&)
{
	return true;
}

//! stream Tag
string MarketMakerAlgorithm::Tag() const
{
	return "";	
}

bool MarketMakerAlgorithm::Dump(ostream&) const
{
	return true;
}

bool MarketMakerAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	return true;
}

Time MarketMakerAlgorithm::CurrentTime() const
{
	return _time;
}

bool go = true;

Rate MarketMakerAlgorithm::CurrentRate() const
{
	return _state[0];
}
