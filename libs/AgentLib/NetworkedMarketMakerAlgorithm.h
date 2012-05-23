#include "DynamicLib/AbstractAlgorithmCode.h"
#include "DynamicLib/DynamicLib.h"
#include "Scheduler.h"
#include <vector>
#include <iostream>

using namespace DynamicLib;
using namespace SparseImplementationLib;
using std::cout;
using std::endl;


class NetworkedMarketMakerAlgorithm : public AbstractAlgorithm<double>
{
	public:
		NetworkedMarketMakerAlgorithm(double, Scheduler);
		~NetworkedMarketMakerAlgorithm();

		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time 
		);

		//! clone
		virtual NetworkedMarketMakerAlgorithm* Clone() const;

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

		virtual double CurrentTime() const; 

		virtual double CurrentRate() const; 

		virtual int GetNextTrader();


	protected:
		
		vector<double> _state;
		
		int _time;

		double _total_demand;
		double _f_demand;
		double _c_demand;

		double _f_price;
		double _c_price;

		bool _is_update_mode;

		Scheduler _sched;
};
