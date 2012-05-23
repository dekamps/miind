#include "DynamicLib/AbstractAlgorithmCode.h"

using namespace DynamicLib;

	class WalrasianAuctioneerAlgorithm : public AbstractAlgorithm<double>
	{
	public:
		WalrasianAuctioneerAlgorithm(const vector<double>&, const string );
		~WalrasianAuctioneerAlgorithm();

		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time 
		);

		//! clone
		virtual WalrasianAuctioneerAlgorithm* Clone() const;

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
		string _tag;

		Time _time;

	};

