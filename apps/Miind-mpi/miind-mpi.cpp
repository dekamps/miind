/*
 * main.cpp
 *
 *  Created on: May 16, 2012
 *      Author: david
 */

#include <boost/mpi.hpp>
#include <iostream>
#include <string>
#include <boost/serialization/string.hpp>
#include <boost/serialization/base_object.hpp>

#include <exception>
#include <MPILib/include/MPINode.hpp>
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/EmptyAlgorithm.hpp>

namespace mpi = boost::mpi;
using namespace MPILib;


int main(int argc, char* argv[]) {
	// initialize mpi

	mpi::environment env(argc, argv);
	try {
		MPINetwork network;
		EmptyAlgorithm alg;


		int node0 = network.AddNode(alg, 1);
		int node1 = network.AddNode(alg, 1);
		int node2 = network.AddNode(alg, 1);

		double weight = 3.1;
		network.MakeFirstInputOfSecond(node0, node1, weight);
		weight = 1.2;
		network.MakeFirstInputOfSecond(node0, node2, weight);

		weight = 6.4;

		network.MakeFirstInputOfSecond(node1, node0, weight);
		weight = 6.1;
		network.MakeFirstInputOfSecond(node2, node1, weight);
		network.Evolve();
	} catch (std::exception & e) {
		std::cout << e.what();
	};

	return 0;
}
