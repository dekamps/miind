#include "ChiarellasModelSimulator.h"
#include "RandSimulator.h"

template <class T>
inline std::string to_string (const T& t)
{
	std::stringstream ss;
	ss << t;
	return ss.str();
}

void ChiarellasModelSimulator::run_convergence_demo_det()
{
	FcmSimulationParameter param;

	run(param);
}

void ChiarellasModelSimulator::run_convergence_demo_rnd()
{
	FcmSimulationParameter param;

	param._no_fundamentalists = 100;
	param._no_chartists = 100;
	param._mode = RANDOM;

	run(param);
}

void ChiarellasModelSimulator::run_convergence_demo_sync()
{
	FcmSimulationParameter param;

	param._mode = SYNCHRONOUS;

	param._price_min = -2.0;
	param._price_max = 3.0;

	run(param);
}



void ChiarellasModelSimulator::run_limit_cycle_demo_det()
{
	FcmSimulationParameter param;

	param._average_speed_c = 0.28;

	run(param);
}

void ChiarellasModelSimulator::run_limit_cycle_demo_rnd()
{
	FcmSimulationParameter param;

	param._no_fundamentalists = 100;
	param._no_chartists = 100;
	param._average_speed_c = 0.28;
	param._mode = RANDOM;

	run(param);
}

void ChiarellasModelSimulator::run_limit_cycle_demo_sync(double init, double b, double c, double a)
{
	FcmSimulationParameter param;

	param._no_fundamentalists = 1;
	param._no_chartists = 1;

	param._average_curvature = b;
	param._average_influence_a = a;
	param._average_speed_c = c;

	param._equilibrium_price = 0.5;
	param._simulation_steps = 2000;


	param._initial_price = init;
	//param._average_trend_length = 4;

	//param._price_max = param._equilibrium_price+(param._equilibrium_price/param._average_influence_a);
	//param._price_min = param._equilibrium_price-(param._equilibrium_price/param._average_influence_a);

	param._price_max = 3;
	param._price_min = -2.0;

	param._mode = SEQUENTIAL;
	param._mode = SYNCHRONOUS;


	param._filename = string("results\\centralized\\sync_det_").append(
		string("_b=").append(to_string(param._average_curvature).append(
		string("_c=").append(to_string(param._average_speed_c).append(
		string("_a=").append(to_string(param._average_influence_a).append(
		string("_eq=").append(to_string(param._equilibrium_price).append(
		string("_in=").append(to_string(param._initial_price).append(
		".root")))))))))));

	param._equilibrium_stable = true;

	run(param);
}



void ChiarellasModelSimulator::run_chaos_demo_det()
{
	FcmSimulationParameter param;

	param._average_speed_c = 5;
	param._average_curvature = 3.0;
	param._average_trend_length = 2;
	param._equilibrium_stable = false;

	run(param);
}
void ChiarellasModelSimulator::run_chaos_demo_rnd()
{
	FcmSimulationParameter param;

	param._no_fundamentalists = 100;
	param._no_chartists = 100;
	param._average_speed_c = 5;
	param._average_curvature = 3.0;
	param._average_trend_length = 2;
	param._equilibrium_stable = false;

	param._mode = RANDOM;

	run(param);
}

void ChiarellasModelSimulator::run_chaos_demo_sync()
{
	FcmSimulationParameter param;

	param._average_speed_c = 5;
	param._average_curvature = 3.0;
	param._average_trend_length = 2;
	param._equilibrium_stable = false;

	param._mode = SYNCHRONOUS;

	run(param);
}

void ChiarellasModelSimulator::run(FcmSimulationParameter param)
{

/**      SIMULATION PARAMETERS      **/

	int no_fundamentalists = param._no_fundamentalists;
	bool influence_a_fixed = param._influence_a_fixed;
	double average_influence_a = param._average_influence_a;
		
	int no_chartists = param._no_chartists;
	bool speed_c_fixed = param._speed_c_fixed;
	double average_speed_c = param._average_speed_c;
	bool curvature_fixed = param._curvature_fixed;
	double average_curvature = param._average_curvature;
	bool trend_length_fixed = param._trend_length_fixed;
	int average_trend_length = param._average_trend_length;

	double equilibrium_price = param._equilibrium_price;
	double initial_price = param._initial_price;
	bool equilibrium_stable = param._equilibrium_stable;

	int mode = param._mode;

	double simulation_steps = param._simulation_steps;

	srand ( param._random_seed );

/*************************************/

	// creating network
	D_DynamicNetwork network;
	network.SetDalesLaw(false);
	Efficacy epsilon = 1;


	// setting up market node
	vector<double> market(3);
	market[0] = initial_price;
	market[1] = equilibrium_price;
	market[2] = 1;
	MarketMakerAlgorithm market_maker(market, mode, equilibrium_price, equilibrium_stable);



	// creating algorithms for nodes
	
	int agent_counter = 1;

	// creating and adding fundamentalists to network
	vector<NodeId> fundamentalists(no_fundamentalists);
	double influence_a;
	for(int i=0; i<no_fundamentalists; i++, agent_counter++)
	{
		if(influence_a_fixed)
			influence_a = average_influence_a;
		else
			influence_a = AgentLib::RandSimulator() % (int)(2*average_influence_a+1);
		fundamentalists[i] = network.AddNode(FundamentalistTraderAlgorithm(agent_counter, influence_a), EXCITATORY);
	}


	// creating and adding chartists to network
	vector<NodeId> chartists(no_chartists);
	int trend_length;

	double curvature, speed_c, total_curvature=0;
	for(int i=0; i<no_chartists; agent_counter++, i++)
	{
		if(trend_length_fixed)
			trend_length = average_trend_length;
		else
			trend_length = AgentLib::RandSimulator() % (int)(2*average_trend_length + 1);


		if(speed_c_fixed)
			speed_c = average_speed_c;
		else
			speed_c = AgentLib::RandSimulator() % (int)(2*average_speed_c+1);

		if(curvature_fixed)
			curvature = average_curvature;
		else
			curvature = AgentLib::RandSimulator() % (int)(2*average_curvature+1);

		total_curvature += curvature;

		chartists[i] = network.AddNode(ChartistTraderAlgorithm(agent_counter, trend_length, curvature, speed_c), EXCITATORY);
	}

	// adding market node to the network
	NodeId idM = network.AddNode(market_maker, EXCITATORY);


	//connecting nodes with the market
	for(int i=0; i<no_fundamentalists; i++)
	{
		network.MakeFirstInputOfSecond(fundamentalists[i], idM, epsilon);
		network.MakeFirstInputOfSecond(idM, fundamentalists[i], epsilon);
	}
	for(int i=0; i<no_chartists; i++)
	{
		network.MakeFirstInputOfSecond(chartists[i], idM, epsilon);
		network.MakeFirstInputOfSecond(idM, chartists[i], epsilon);
	}

	// define a handler to store the simulation results
	CMRootReportHandler 
		handler
		(
			"test/"+param._filename,	// file where the simulation results are written
			true,							// display on screen
			true							// write into file
		);

	// defining simulation parameters
	SimulationRunParameter
		par_run
		(
			handler,					// the handler object
			0,							// maximum number of iterations
			0,							// start time of simulation
			simulation_steps,			// end time of simulation
			1,							// report time
			1,							// update time
			1,							// network step time
			"test/fcmresponse.log"   // log file name
		);


	// decide the ranges to show in the canvas
	handler.SetFrequencyRange(param._price_min, param._price_max);
	handler.SetTimeRange(0.,simulation_steps);
	
	handler.AddNodeToCanvas(idM);     // we want to visualize only the market maker

	bool b_configure = network.ConfigureSimulation(par_run);

	if (! b_configure)
		return;

	bool b_evolve = network.Evolve();
	if (! b_evolve)
		return;
}
