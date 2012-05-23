#include "AbstractNetworkTraderAlgorithm.h"

class FundamentalistNetworkTraderAlgorithm:
	public AbstractNetworkTraderAlgorithm
{
	public:
		FundamentalistNetworkTraderAlgorithm(int, double, double, double);
		~FundamentalistNetworkTraderAlgorithm(void);

		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time 
		);

		//! clone
		virtual FundamentalistNetworkTraderAlgorithm* Clone() const;

		//!
		virtual string LogString() const;


	private:

		double		_a;
		double		_W;
};
