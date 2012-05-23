#include "NetworkedModelSimulator.h"


void NetworkedModelSimulator::RunStar(networked_simulation_parameter param)
{
	// creating network
	D_DynamicNetwork network;
	network.SetDalesLaw(false);
	Efficacy epsilon = 1;


	// reading fundamentalists and chartists from file
	ifstream indata; // indata is like cin

	indata.open(param.in_filename.c_str()); // opens the file
	if(!indata) { // file couldn't be opened
		cerr << "Error: file could not be opened" << endl;
		exit(1);
	}

	int f_nodes, c_nodes, id; // variable for input value
	char type;

	indata >> f_nodes;
	indata >> c_nodes;

	vector<NodeId> funds(f_nodes), charts(c_nodes);

	double initial_price = 0.0;

	// reading nodes and adding connecting them with market maker
	for(int i=0; i<f_nodes+c_nodes; i++)
	{
		indata >> type;
		indata >> id;
		if(type == 'c') 
		{
			charts[id] = network.AddNode(ChartistNetworkTraderAlgorithm(id, param.b, param.c, param.initial_price), EXCITATORY);
			//network.MakeFirstInputOfSecond(idM, charts[id], epsilon);
			//network.MakeFirstInputOfSecond(charts[id], idM, epsilon);
		}
		else
		{
			funds[id] = network.AddNode(FundamentalistNetworkTraderAlgorithm(id, param.a, param.W, param.initial_price), EXCITATORY);
			//network.MakeFirstInputOfSecond(idM, funds[id], epsilon);
			//network.MakeFirstInputOfSecond(funds[id], idM, epsilon);
		}
	}

	// creating scheduler

	// setting up market node
	NetworkedMarketMakerAlgorithm market_maker(1.0, Scheduler(f_nodes, c_nodes, param._type));

	// adding market node to the network
	NodeId idM = network.AddNode(market_maker, EXCITATORY);

	for(int i=0; i<f_nodes; i++)
	{
		network.MakeFirstInputOfSecond(idM, funds[i], epsilon);
		network.MakeFirstInputOfSecond(funds[i], idM, epsilon);

		network.MakeFirstInputOfSecond(idM, charts[i], epsilon);
		network.MakeFirstInputOfSecond(charts[i], idM, epsilon);
	}

	int edges;
	NodeId id1, id2;

	indata >> edges;

	for(int i=0; i < edges; i++)
	{
		indata >> type;
		indata >> id;
		if(type == 'f')
			id1 = funds[id];
		else
			id1 = charts[id];
		
		indata >> type;
		
		indata >> type;
		indata >> id;
		if(type == 'f')
			id2 = funds[id];
		else
			id2 = charts[id];

		network.MakeFirstInputOfSecond(id1, id2, epsilon);
		network.MakeFirstInputOfSecond(id2, id1, epsilon);
	}

	indata.close();


	// define a handler to store the simulation results
	CMRootReportHandler 
		handler
		(
			"test/"+param.out_filename,	// file where the simulation results are written
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
			param.simulation_steps,			// end time of simulation
			1,							// report time
			1,							// update time
			1,							// network step time
			"test/fcmresponse.log"   // log file name
		);


	// decide the ranges to show in the canvas
	handler.SetFrequencyRange(-1, 1);
	handler.SetTimeRange(0.,param.simulation_steps);
	
	handler.AddNodeToCanvas(idM);     // we want to visualize only the market maker

	bool b_configure = network.ConfigureSimulation(par_run);

	if (! b_configure)
		return;

	bool b_evolve = network.Evolve();
	if (! b_evolve)
		return;
}