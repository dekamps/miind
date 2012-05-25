/*
 * MPINode.h
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#ifndef MPINODE_H_
#define MPINODE_H_

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

namespace mpi = boost::mpi;

class MPINode {
public:
// asynchronous send method
	template<typename T>
	mpi::request send(T s, int dest, int tag, mpi::communicator w) {
		return w.isend(dest, tag, s);
	}

// asynchronous recv method
	template<typename T>
	mpi::request recv(T& s, int origin, int tag, mpi::communicator w) {
		return w.irecv(origin, tag, s);
	}
};

#endif /* MPINODE_H_ */
