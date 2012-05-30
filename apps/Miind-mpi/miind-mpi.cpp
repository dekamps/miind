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

namespace mpi = boost::mpi;
using namespace MPILib;



//make it mpi datatype the class with std::string this is not possible as it is no MPI Datatype
BOOST_IS_MPI_DATATYPE(StructPoint)

int main(int argc, char* argv[]) {
	// initialize mpi

	mpi::environment env(argc, argv);
	try {
		MPINetwork network;

		int node0 = network.AddNode(1, 1);
		int node1 = network.AddNode(1, 1);
		int node2 = network.AddNode(1, 1);
		int node3 = network.AddNode(1, 1);
		int node4 = network.AddNode(1, 1);
		int node5 = network.AddNode(1, 1);
		int node6 = network.AddNode(1, 1);
		int node7 = network.AddNode(1, 1);
		int node8 = network.AddNode(1, 1);
		int node9 = network.AddNode(1, 1);
		int node10 = network.AddNode(1, 1);
		double weight = 3.1;
		network.MakeFirstInputOfSecond(node0, node1, weight);
		network.MakeFirstInputOfSecond(node1, node2, weight);
		network.MakeFirstInputOfSecond(node2, node3, weight);
		network.MakeFirstInputOfSecond(node3, node4, weight);
		network.MakeFirstInputOfSecond(node4, node5, weight);
		network.MakeFirstInputOfSecond(node5, node6, weight);
		network.MakeFirstInputOfSecond(node6, node7, weight);
		network.MakeFirstInputOfSecond(node7, node8, weight);
		network.MakeFirstInputOfSecond(node8, node9, weight);
		network.MakeFirstInputOfSecond(node9, node10, weight);
		network.MakeFirstInputOfSecond(node10, node0, weight);

		network.Evolve();
	} catch (std::exception & e) {
		std::cout << e.what();
	};

	return 0;
}
