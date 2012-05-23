#include "AbstractNetworkTraderAlgorithm.h"
#include "Scheduler.h"

AbstractNetworkTraderAlgorithm::AbstractNetworkTraderAlgorithm()
:AbstractAlgorithm<double>(1)
{
		_state = vector<double>(5);
		//_state[0] = -1;
		_state[1] = 0;
		_state[2] = 0;
		_state[3] = 1;
		_new_price = 0;
		_is_update_mode = false;

}

AbstractNetworkTraderAlgorithm::~AbstractNetworkTraderAlgorithm(void)
{
}

//! evolve node state
bool AbstractNetworkTraderAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{
	_time = time;
	
	//if(_is_update_mode)
	//	_time --;

	//if(_is_update_mode){
		// retrieving data from traders
		D_AbstractSparseNode* p_node;
		D_DynamicNode* p_dyn;

		int trader_id;

		double f_demand = 0;
		double c_demand = 0;
		int c_counter = 0;
		int f_counter = 0;
		predecessor_iterator cii;
		for(cii=begin_iter; cii!=end_iter; cii++)
		{
			p_node = &(*cii); 
			p_dyn = dynamic_cast<D_DynamicNode*>(p_node);	

			if (p_dyn)
			{
				auto_ptr<AbstractAlgorithm<double> > algorithm = p_dyn->CloneAlgorithm();
				if(algorithm->State()[0] < 0)
				{
					if(algorithm->State()[0] == -1)
					{
						f_counter++;
						f_demand += algorithm->State()[1];
					}
					else
					{
						c_counter++;
						c_demand += algorithm->State()[1];
					}
				}
				else
					{
						trader_id = algorithm->State()[4];
						
						// I'm not scheduled this round..
						if(!((_state[0]==-1 && trader_id == Scheduler::f+_id) || (_state[0]==-2 && trader_id == Scheduler::c+_id)) )
						{
							return false;
						}
						
					}
			}
		}

		if(_state[0] == -1)
		{
			f_counter++;
			f_demand += _state[1];
		}
		else
		{
			c_counter++;
			c_demand += _state[1];
		}

		if(c_counter != 0)
			c_demand /= c_counter;
		if(f_counter != 0)
			f_demand /= f_counter;

		_state[2] += f_demand + c_demand;

		//cout << _state[0] << " " <<_state[2] << endl;
		//_state[2] += f_demand/f_counter + c_demand/c_counter;
//
	//}

	//_is_update_mode = !_is_update_mode;

	return true;		
}

//! report node state
NodeState AbstractNetworkTraderAlgorithm::State() const
{
	NodeState ns(_state);
	return ns;
}

//! streaming to output stream
bool AbstractNetworkTraderAlgorithm::ToStream(ostream&) const
{
	return true;
}

//! streaming from input stream
bool AbstractNetworkTraderAlgorithm::FromStream(istream&)
{
	return true;
}

//! stream Tag
string AbstractNetworkTraderAlgorithm::Tag() const
{
	return "";	
}

bool AbstractNetworkTraderAlgorithm::Dump(ostream&) const
{
	return true;
}

bool AbstractNetworkTraderAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	return true;
}

Time AbstractNetworkTraderAlgorithm::CurrentTime() const
{
	return _time;
}

Rate AbstractNetworkTraderAlgorithm::CurrentRate() const
{
	return _state[0];
}



//! report initial state
AlgorithmGrid AbstractNetworkTraderAlgorithm::Grid() const
{
	return _state;
}
