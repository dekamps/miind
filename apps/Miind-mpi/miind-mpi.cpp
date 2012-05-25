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

//// asynchronous send method
//template<typename T>
//mpi::request send(T s, int dest, int tag, mpi::communicator w) {
//	return w.isend(dest, tag, s);
//}
//
//// asynchronous recv method
//template<typename T>
//mpi::request recv(T& s, int origin, int tag, mpi::communicator w) {
//	return w.irecv(origin, tag, s);
//}

int main(int argc, char* argv[]) {
	// initialize mpi

	mpi::environment env(argc, argv);

	mpi::communicator world;

	if (world.rank() == 0) {
		//store asynchronous requests
		mpi::request reqs[2];

		StuctPointMessage p;
		p.x = 1.;
		p.y = 2;
		p.z = 3.;
		p.msg = "blub";

		std::string msg;
		MPINode blub;
		// the order of send and recv is unimportant as they are asynchronous
		// with synchronous this would result in a deadlock
		reqs[1] = blub.recv(msg, 1, 1, world);
		reqs[0] = blub.send(p, 1, 2, world);
		// make sure the sended data has arrived
		reqs[1].wait();
		std::cout << msg << "!" << std::endl;

	} else {
		mpi::request reqs[2];
		MPINode blub;

		StuctPointMessage p;
		reqs[0] = blub.recv(p, 0, 2, world);
		reqs[1] = blub.send(std::string("recieved"), 0, 1, world);

		reqs[0].wait();
		p.print();

	}

	return 0;
}
