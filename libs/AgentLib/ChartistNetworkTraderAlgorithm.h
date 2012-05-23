#include "AbstractNetworkTraderAlgorithm.h"

class ChartistNetworkTraderAlgorithm :
	public AbstractNetworkTraderAlgorithm
{
	public:
		ChartistNetworkTraderAlgorithm(int, double, double, double);
		~ChartistNetworkTraderAlgorithm(void);

		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time 
		);

		//! clone
		virtual ChartistNetworkTraderAlgorithm* Clone() const;

		//!
		virtual string LogString() const;



	private:

		double		_b;
		double		_c;

		double		_last_price;
		double		_estimate;
		int			_action_counter;

};
