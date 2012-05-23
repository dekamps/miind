
#include "ChartistTraderAlgorithm.h"

ChartistTraderAlgorithm::ChartistTraderAlgorithm
(	
	const int id,
	const int trend_length,
	const double curvature,
	const double speed_c
):
 AbstractAlgorithm<double>(1)
{
	_id = id;
	_state = vector<double>(5);
	_time = 0;
	_curvature = curvature;
	_speed_c = speed_c;
	_last_price = 0;
	_action_counter = 0;
	_alpha = 0.5;
	_estimate = 0;
}

ChartistTraderAlgorithm::~ChartistTraderAlgorithm(void)
{
}

//! evolve node state
bool ChartistTraderAlgorithm::EvolveNodeState
(
	predecessor_iterator begin_iter,
	predecessor_iterator end_iter,
	Time time
)
{
	_time = time;

	D_AbstractSparseNode* p_node = &(*begin_iter); 
	D_DynamicNode* p_dyn = dynamic_cast<D_DynamicNode*>(p_node);

	if (p_dyn)
	{
		auto_ptr<AbstractAlgorithm<double> > algorithm = p_dyn->CloneAlgorithm();
		double current_price = algorithm->State()[0];

		double price_change = 0;
		if(_action_counter++ > 1)
			price_change = current_price - _last_price;

		_last_price = current_price;

		_estimate = _estimate + _speed_c * (price_change-_estimate);

		// calculating total

		if(algorithm->State()[2] == _id || algorithm->State()[2] == -1){
			_state[0] = (1/(1+exp(-4*_curvature*_estimate))-0.5);
		}
		else
			_state[0] = 0;

		_state[1] = _estimate;

	}

	return true;		
}


//! clone
ChartistTraderAlgorithm* ChartistTraderAlgorithm::Clone() const
{
	return new ChartistTraderAlgorithm(*this);
}

//! report initial state
AlgorithmGrid ChartistTraderAlgorithm::Grid() const
{
	return _state;
}

//! report node state
NodeState ChartistTraderAlgorithm::State() const
{
	NodeState ns(_state);
	return ns;
}

//!
string ChartistTraderAlgorithm::LogString() const 
{
	return "c 0 0";
}

//! streaming to output stream
bool ChartistTraderAlgorithm::ToStream(ostream&) const
{
	return true;
}

//! streaming from input stream
bool ChartistTraderAlgorithm::FromStream(istream&)
{
	return true;
}

//! stream Tag
string ChartistTraderAlgorithm::Tag() const
{
	return "chartist";	
}

bool ChartistTraderAlgorithm::Dump(ostream&) const
{
	return true;
}

bool ChartistTraderAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	return true;
}

Time ChartistTraderAlgorithm::CurrentTime() const
{
	return _time;
}

Rate ChartistTraderAlgorithm::CurrentRate() const
{
	return _state[0];
}
