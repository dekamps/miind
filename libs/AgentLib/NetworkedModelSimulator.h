#include <string>
#include <stdlib.h>
#include <sstream>
#include <cstdlib>
#include <fstream>

#include "ChartistNetworkTraderAlgorithm.h"
#include "FundamentalistNetworkTraderAlgorithm.h"
#include "CMRootReportHandler.h"
#include "NetworkedMarketMakerAlgorithm.h"
#include "Scheduler.h"

using std::ifstream;

using namespace DynamicLib;

#define FULLY_CONNECTED 1
#define LOCAL_CONNECTED 2
#define SMALL_WORLD 3
#define SPHERE 3

template <class T>
inline std::string to_string (const T& t)
{
	std::stringstream ss;
	ss << t;
	return ss.str();
}

struct networked_simulation_parameter
{
	enum SimulationType {INTERLEAVED_SEQUENTIAL, RANDOM_INTERLEAVED, COMPLETE_RANDOM };

	networked_simulation_parameter(const string& nettype):
		a(0.1),
		c(0.1),
		b(1.7),
		initial_price(0.505),
		W(0.5),
		equilibrium_stable(true),
		simulation_steps(2000),
		out_filename("test.root"),
		price_min(-1),
		price_max(1),
		_type(INTERLEAVED_SEQUENTIAL),
		in_filename("../../../apps/Trader/networks/"+ nettype +".txt")
	{
		out_filename = string("results/networked/16-as_fully_connected_").append(
		string("_b=").append(to_string(b).append(
		string("_c=").append(to_string(c).append(
		string("_a=").append(to_string(a).append(
		string("_eq=").append(to_string(W).append(
		string("_in=").append(to_string(initial_price).append(
		".root")))))))))));		
	};

	double a;
		
	double c;
	double b;

	double initial_price;

	double W;
	bool equilibrium_stable;

	double simulation_steps;

	string out_filename;
	double price_min;
	double price_max;

	SimulationType _type;
	string in_filename;
};

class NetworkedModelSimulator
	{
	public:
		virtual void RunStar(networked_simulation_parameter);
	};
