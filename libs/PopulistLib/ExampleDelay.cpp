#include "../DynamicLib/DynamicLib.h"
#include "TestDefinitions.h"
#include "TestBinCalculationDefinitions.h"
#include "TestResponseCurveDefinitions.h"
#include "OrnsteinUhlenbeckAlgorithm.h"

using DynamicLib::EXCITATORY;
using DynamicLib::INHIBITORY;
/*
namespace PopulistLib {

// Note we now need an OU_Network instead of a D_Network
	OU_Network net;

	Potential sigma = 2e-3;
	Potential mu    = 20e-3;

	Time tau = PARAMETER_NEURON._tau; 
	Rate nu = mu*mu/(sigma*sigma*tau);
	Rate J = sigma*sigma/mu;

	OU_Connection 
		con
		(
			1,
			J
		);

	// Define a node with a fixed output rate
	OU_RateAlgorithm rate_alg(nu);
	NodeId id_rate = net.AddNode(rate_alg,EXCITATORY);

	// Define the receiving node 
	OU_Algorithm algorithm_exc(PARAMETER_NEURON);
	NodeId id = net.AddNode(algorithm_exc,EXCITATORY);

	// connect the two nodes
	net.MakeFirstInputOfSecond(id_rate,id,con);

	// define a handler to store the simulation results
	RootReportHandler 
		handler
		(
			"test/ouresponse.root",	// simulation results 
			false,		// do not display on screen
			true		// write into file
		);

	SimulationRunParameter
		par_run
		(
			handler,	// the handler object
			1000000,	// maximum number of iterations
			0,		// start time of simulation
			0.1,		// end time of simulation
			1e-4,		// report time
			1e-4,		// update time
			1e-5,		// network step time
			"test/ouresponse.log"   // log file name
		);

	bool b_configure = net.ConfigureSimulation(par_run);

	bool b_evolve = net.Evolve();
	// ending
}
*/