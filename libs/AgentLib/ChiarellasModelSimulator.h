#include <string>
#include <stdlib.h>
#include <sstream>
//#include "DynamicLib/DynamicLibTest.h"
#include "MarketMakerAlgorithm.h"
#include "ChartistTraderAlgorithm.h"
#include "FundamentalistTraderAlgorithm.h"
#include "CMRootReportHandler.h"
//#include "DynamicLib/RootReportHandler.h"

using namespace DynamicLib;

// simulation paremeter set
struct FcmSimulationParameter
{

	FcmSimulationParameter() : 
		_no_fundamentalists(1),
		_influence_a_fixed(true),
		_average_influence_a(0.15),
			
		_no_chartists(1),
		_speed_c_fixed(true),
		_average_speed_c(0.2),
		_curvature_fixed(true),
		_average_curvature(2.0),
		_trend_length_fixed(true),
		_average_trend_length(20),

		_equilibrium_price(0.5),
		_initial_price(0.7),
		_equilibrium_stable(true),

		_mode(SEQUENTIAL),
		_simulation_steps(1000),
		_price_min(-0.5),
		_price_max(1.5),

		_random_seed(666),

		_filename("fcm.root")
	{}

	int _no_fundamentalists;		// number of fundamentalists in the model
	bool _influence_a_fixed;		// define whether influence a is fixed 
	double _average_influence_a;	// average size of influence parameter a
	
	int _no_chartists;				// number of chartist in the model
	bool _speed_c_fixed;			// speed of price adjustment
	double _average_speed_c;		// average price adjustment speed	
	bool _curvature_fixed;			// define whether steepness of price adjustment is fixed
	double _average_curvature;		// average steepness parameter size
	bool _trend_length_fixed;		// defines whether trend length is fixed
	int _average_trend_length;		// average trend length

	double _equilibrium_price;		// equilibrium price
	double _initial_price;			// initial starting price
	bool _equilibrium_stable;		// defines whether equilibrium is stable or not

	int _mode;						// defines the mode of simulation execution

	double _simulation_steps;		// number of simulation steps
	double _price_min;				// displayed price range min
	double _price_max;				// displayed price range max

	long _random_seed;				// seed for the random number generator
	string _filename;				// name for the output file

};

// model simulator class
class ChiarellasModelSimulator {
	
public:

	virtual void run_convergence_demo_det();
	virtual void run_convergence_demo_rnd();
	virtual void run_convergence_demo_sync();

	virtual void run_limit_cycle_demo_det();
	virtual void run_limit_cycle_demo_rnd();
	virtual void run_limit_cycle_demo_sync(double init, double b, double c, double a);


	virtual void run_chaos_demo_det();
	virtual void run_chaos_demo_rnd();
	virtual void run_chaos_demo_sync();

	virtual void run(FcmSimulationParameter);
	
};

