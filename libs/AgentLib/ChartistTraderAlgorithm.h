#include "DynamicLib/AbstractAlgorithmCode.h"
#include "DynamicLib/DynamicLib.h"
#include <vector>
#include <iostream>

using namespace DynamicLib;
using namespace SparseImplementationLib;
using std::cout;
using std::endl;


class ChartistTraderAlgorithm : public AbstractAlgorithm<double>
{
public:
	ChartistTraderAlgorithm( const int, const int, const double, const double );
	~ChartistTraderAlgorithm();

	virtual bool EvolveNodeState
	(
		predecessor_iterator,
		predecessor_iterator,
		Time 
	);

	//! clone
	virtual ChartistTraderAlgorithm* Clone() const;

	//! report initial state
	virtual AlgorithmGrid Grid() const;

	//! report node state
	virtual NodeState State() const;

	//!
	virtual string LogString() const;

	//! streaming to output stream
	virtual bool ToStream(ostream&) const;

	//! streaming from input stream
	virtual bool FromStream(istream&);

	//! stream Tag
	string Tag() const;

	virtual bool Dump(ostream&) const;

	virtual bool Configure
	(
		const SimulationRunParameter&
	);

	virtual Time CurrentTime() const; 

	virtual Rate CurrentRate() const;

private:

	/*
	[current_demand, previous_demand]
	*/
	vector<double> _state;

	/*
		last price
	*/
	double _last_price;

	double _curvature;
	double _speed_c;
	double _estimate;
	double _alpha;

	int _action_counter;

	int _id;

	Time _time;

};

