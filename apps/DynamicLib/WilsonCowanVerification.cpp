/*
 * wilsonCowanVerification.cpp
 *
 *  Created on: 11 Jun 2012
 *      Author: david
 */

#include <DynamicLib.h>
#include <DynamicLib/DynamicNetworkCode.h>
#include <DynamicLib/DynamicNetworkImplementationCode.h>
#include <DynamicLib/WilsonCowanAlgorithm.h>
#include <DynamicLib/WilsonCowanParameter.h>
#include <DynamicLib/TestDefinitions.h>

using namespace DynamicLib;

int main(int argc, char* argv[]) {

// define a D_Network, a network whose weights are doubles
	D_DynamicNetwork network_wctest;

	Time tau = 10e-3; //10 ms
	Rate rate_max = 100.0;
	double noise = 1.0;

// define some efficacy
	Efficacy epsilon = 1.0;

// define some input rate
	Rate nu = 0;


// Define a node with a fixed output rate
	D_RateAlgorithm rate_alg(nu);
	NodeId id_rate = network_wctest.AddNode(rate_alg, EXCITATORY);

// Define the receiving node
	WilsonCowanParameter par_sigmoid(tau, rate_max, noise);

	WilsonCowanAlgorithm algorithm_exc(par_sigmoid);
	NodeId id = network_wctest.AddNode(algorithm_exc, EXCITATORY);

// connect the two nodes
	network_wctest.MakeFirstInputOfSecond(id_rate, id, epsilon);

	bool b_configure = network_wctest.ConfigureSimulation(PAR_WILSONCOWAN);

	if (!b_configure)
		return false;

	bool b_evolve = network_wctest.Evolve();
	if (!b_evolve)
		return false;

// ending
}
