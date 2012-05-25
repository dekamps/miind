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

#include <MPILib/MPINode.hpp>
#include <MPILib/MPINetwork.hpp>

namespace mpi = boost::mpi;

//some user defined data to send around
class StructPoint {
public:
	double x;
	int y;
	double z;
	void print() {
		std::cout << x << "\t" << y << "\t" << z << std::endl;
	}
	//make it serialized to send it around
	friend class boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & x;
		ar & y;
		ar & z;
	}
	virtual ~StructPoint() {
	}
	;

};
//derived class to show concept
class StuctPointMessage: public StructPoint {
public:
	friend class boost::serialization::access;

	std::string msg;
	void print() {
		std::cout << x << "\t" << y << "\t" << z << "\t" << msg << std::endl;
	}
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		//call the base class serialization
		ar & boost::serialization::base_object<StructPoint>(*this);
		ar & msg;
	}

};

//make it mpi datatype the class with std::string this is not possible as it is no MPI Datatype
BOOST_IS_MPI_DATATYPE(StructPoint)



int main(int argc, char* argv[]) {
	// initialize mpi

	mpi::environment env(argc, argv);

	MPINetwork network;

	network.AddNode(1,1,0);
	network.AddNode(1,1,1);
	network.AddNode(1,1,2);
	network.AddNode(1,1,3);
	network.AddNode(1,1,4);
	network.AddNode(1,1,5);
	network.AddNode(1,1,6);
	network.AddNode(1,1,7);
	network.AddNode(1,1,8);
	network.Evolve();

	return 0;
}
