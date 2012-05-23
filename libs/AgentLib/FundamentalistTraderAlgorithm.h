#include "DynamicLib/AbstractAlgorithmCode.h"
#include "DynamicLib/DynamicLib.h"


#include <vector>
#include <iostream>

using namespace DynamicLib;
using namespace SparseImplementationLib;
using std::cout;
using std::endl;


class FundamentalistTraderAlgorithm : public AbstractAlgorithm<double>
{
public:
	FundamentalistTraderAlgorithm( const int, const double );
	~FundamentalistTraderAlgorithm();

	virtual bool EvolveNodeState
	(
		predecessor_iterator,
		predecessor_iterator,
		Time 
	);

	//! clone
	virtual FundamentalistTraderAlgorithm* Clone() const;

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

	vector<double> _state;

	int _id;

	double _influence_a;

	Time _time;

};

