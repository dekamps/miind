#include "DynamicLib/AbstractAlgorithmCode.h"
#include "DynamicLib/DynamicLib.h"
#include "WalrasianTraderAlgorithm.h"
#include "gsl/gsl_rng.h"
#include <vector>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <time.h>

using namespace DynamicLib;
using namespace SparseImplementationLib;
using std::cout;
using std::endl;


// simulation execution modes
#define RANDOM -11		// random ordering of agents
#define SEQUENTIAL -22	// sequential ordering - first all funds then all charts
#define SYNCHRONOUS -33	// all update simultaneously

class MarketMakerAlgorithm : public AbstractAlgorithm<double>
{
public:
	MarketMakerAlgorithm(const vector<double>&, const int, const double, const bool );
	~MarketMakerAlgorithm();

	virtual bool EvolveNodeState
	(
		predecessor_iterator,
		predecessor_iterator,
		Time 
	);

	//! clone
	virtual MarketMakerAlgorithm* Clone() const;

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

	int _mode;
	
	bool _equilibrium_stable;

	Time _time;

	double _psi, _cd, _fd;

	bool _is_update_mode;

};

