#include "NetworkedMarketMakerAlgorithm.h"


NetworkedMarketMakerAlgorithm::NetworkedMarketMakerAlgorithm(double d, Scheduler sched) : AbstractAlgorithm<double>(d)
{
		_state = vector<double>(5);
		_total_demand = 0;
		_is_update_mode = false;
		_f_demand = 0;
		_c_demand = 0;
		_c_price = 0;
		_f_price = 0;
		_sched = sched;
}

NetworkedMarketMakerAlgorithm::~NetworkedMarketMakerAlgorithm(void)
{
}

//! evolve node state
bool NetworkedMarketMakerAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{
	_time = time;


	// retrieving data from traders
	D_AbstractSparseNode* p_node;
	D_DynamicNode* p_dyn;

	int counter = 0;

	_total_demand = 0;
	_f_demand = 0;
	_c_demand = 0;
	_f_price = 0;
	_c_price = 0;
	predecessor_iterator cii;
	for(cii=begin_iter; cii!=end_iter; cii++)
	{
		p_node = &(*cii); 
		p_dyn = dynamic_cast<D_DynamicNode*>(p_node);	

		if (p_dyn)
		{
			auto_ptr<AbstractAlgorithm<double> > algorithm = p_dyn->CloneAlgorithm();
			
			//cout << "TYPE: " << algorithm->State()[0]  << endl;

			if(algorithm->State()[0] == -1)
			{
				_f_demand += algorithm->State()[1];
				_f_price += algorithm->State()[2];
				
			}
			if(algorithm->State()[0] == -2)
			{			
				_c_demand += algorithm->State()[1];
				_c_price += algorithm->State()[2];
			}
			_total_demand += algorithm->State()[2];
			counter++;
		}
	}

	//cout << "trends: " << _f_demand << " " << _c_demand << endl;
	

	_total_demand /= (double) counter;

	_is_update_mode = !_is_update_mode;

	_state[4] = GetNextTrader();

#ifdef DIAG
	cout << "Scheduling trade no.. " << _state[4] << endl;
#endif

	return true;		
}

int NetworkedMarketMakerAlgorithm::GetNextTrader()
	{
		return _sched.GetTraderId();
	}

//! report node state
NodeState NetworkedMarketMakerAlgorithm::State() const
{
	NodeState ns(_state);
	return ns;
}

//! streaming to output stream
bool NetworkedMarketMakerAlgorithm::ToStream(ostream&) const
{
	return true;
}

//! streaming from input stream
bool NetworkedMarketMakerAlgorithm::FromStream(istream&)
{
	return true;
}

//! stream Tag
string NetworkedMarketMakerAlgorithm::Tag() const
{
	return "";	
}

bool NetworkedMarketMakerAlgorithm::Dump(ostream&) const
{
	return true;
}

bool NetworkedMarketMakerAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	return true;
}

Time NetworkedMarketMakerAlgorithm::CurrentTime() const
{
	return _time;
}

Rate NetworkedMarketMakerAlgorithm::CurrentRate() const
{
	return _total_demand;
}

//! report initial state
AlgorithmGrid NetworkedMarketMakerAlgorithm::Grid() const
{
	return _state;
}


string NetworkedMarketMakerAlgorithm::LogString() const
{
	char s[1000];
	sprintf ( s, "m%f %f %f %f", 0.5, _f_price, _f_demand, _c_demand);
	return s;
}

NetworkedMarketMakerAlgorithm* NetworkedMarketMakerAlgorithm::Clone() const
{
	return new NetworkedMarketMakerAlgorithm(*this);
}
