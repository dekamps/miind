#include "ChartistNetworkTraderAlgorithm.h"

ChartistNetworkTraderAlgorithm::ChartistNetworkTraderAlgorithm(int id, double b, double c, double init):AbstractNetworkTraderAlgorithm()
{
	_id = id;
	_b = b;
	_c = c;
	_action_counter = 0;
	_estimate = 0;
	_last_price = 0;
	_state[0] = -2;
	_state[2] = init;
}

ChartistNetworkTraderAlgorithm::~ChartistNetworkTraderAlgorithm(void)
	{
	}

//! evolve node state
bool ChartistNetworkTraderAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{

	if(!AbstractNetworkTraderAlgorithm::EvolveNodeState(begin_iter, end_iter, time)) {_time = time; return true;}

		
	//if(_is_update_mode){		
		double current_price = _state[2];

		double price_change = 0;
		if(_action_counter++ > 1)
			price_change = current_price - _last_price - _state[1];

		if(price_change != 0)
			price_change += _state[1];

		_last_price = current_price;

		//if(_new_price == price_change)
		//	_estimate = 0;
		//else
			_estimate = _estimate + _c * (price_change-_estimate);

		// calculating total
		_new_price = (1/(1+exp(-4*_b*_estimate))-0.5);
	
	//}else
		_state[1] = _new_price;
#ifdef DIAG
	cout << "c" << _id << " " <<_estimate << " " << _state[1] << " " << _state[2] << endl;
#endif
	_is_update_mode = !_is_update_mode;

	return true;		
}

//! clone
ChartistNetworkTraderAlgorithm* ChartistNetworkTraderAlgorithm::Clone() const
{
	return new ChartistNetworkTraderAlgorithm(*this);
}

//!
string ChartistNetworkTraderAlgorithm::LogString() const 
{
	return "c 0 0";
}