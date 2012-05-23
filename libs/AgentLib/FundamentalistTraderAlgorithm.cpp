
#include "FundamentalistTraderAlgorithm.h"

FundamentalistTraderAlgorithm::FundamentalistTraderAlgorithm
(	
	int id,
	double influence_a
):
AbstractAlgorithm<double>(1)
{
	_id = id;
	_state = vector<double>(2);
	_time = 0;
	_influence_a = influence_a;
}

FundamentalistTraderAlgorithm::~FundamentalistTraderAlgorithm(void)
{
}

//! evolve node state
bool FundamentalistTraderAlgorithm::EvolveNodeState
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
		NodeState vec = algorithm->State();

		if(vec[2] == _id || vec[2] == -1){
			_state[0] = _influence_a*((vec[1] - vec[0]));
			//cout << " f " << _state[0];
		}
		else
			_state[0] = 0;
		_state[1]=-1;
	}

	return true;		
}

//! clone
FundamentalistTraderAlgorithm* FundamentalistTraderAlgorithm::Clone() const
{
	return new FundamentalistTraderAlgorithm(*this);
}

//! report initial state
AlgorithmGrid FundamentalistTraderAlgorithm::Grid() const
{
	return _state;
}

//! report node state
NodeState FundamentalistTraderAlgorithm::State() const
{
	NodeState ns(_state);
	return ns;
}

//!
string FundamentalistTraderAlgorithm::LogString() const 
{
	return "f 0 0";
}

//! streaming to output stream
bool FundamentalistTraderAlgorithm::ToStream(ostream&) const
{
	return true;
}

//! streaming from input stream
bool FundamentalistTraderAlgorithm::FromStream(istream&)
{
	return true;
}

//! stream Tag
string FundamentalistTraderAlgorithm::Tag() const
{
	return "fundamentalist";	
}

bool FundamentalistTraderAlgorithm::Dump(ostream&) const
{
	return true;
}

bool FundamentalistTraderAlgorithm::Configure
(
	const SimulationRunParameter& parameter_simulation
)
{
	return true;
}

Time FundamentalistTraderAlgorithm::CurrentTime() const
{
	return _time;
}

Rate FundamentalistTraderAlgorithm::CurrentRate() const
{
	return _state[0];
}
