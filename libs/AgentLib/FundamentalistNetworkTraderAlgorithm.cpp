#include "FundamentalistNetworkTraderAlgorithm.h"

FundamentalistNetworkTraderAlgorithm::FundamentalistNetworkTraderAlgorithm(int id, double a, double W, double init):AbstractNetworkTraderAlgorithm()
{
	_id = id;
	_a = a;
	_W = W;
	_state[0] = -1;
	_state[2] = init;
}

FundamentalistNetworkTraderAlgorithm::~FundamentalistNetworkTraderAlgorithm(void)
	{
	}


//! evolve node state
bool FundamentalistNetworkTraderAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{

	if(!AbstractNetworkTraderAlgorithm::EvolveNodeState(begin_iter, end_iter, time)) {_time = time; return true;}

	if(_is_update_mode)
		_new_price = _a*(_W - _state[2]);
	else
		_state[1] = _new_price;
	
#ifdef DIAG	
	cout << "f" << _id << " " << _state[1] << " " << _state[2] << endl;
#endif
	_is_update_mode = !_is_update_mode;

	return true;		
}

//! clone
FundamentalistNetworkTraderAlgorithm* FundamentalistNetworkTraderAlgorithm::Clone() const
{
	return new FundamentalistNetworkTraderAlgorithm(*this);
}

//!
string FundamentalistNetworkTraderAlgorithm::LogString() const 
{
	return "f 0 0";
}
