/*
 * MPIProxy.cpp
 *
 *  Created on: 10.07.2012
 *      Author: david
 */

#include <MPILib/config.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
namespace mpi = boost::mpi;
mpi::communicator world;
#endif



namespace MPILib {
namespace utilities {

int MPIProxy::_rank = 0;
int MPIProxy::_size = 1;

MPIProxy::MPIProxy() {
#ifdef ENABLE_MPI
	mpi::communicator world;
	_rank = world.rank();
	_size = world.size();
#endif
}

MPIProxy::~MPIProxy() {
}

int MPIProxy::getRank() const{
	return  _rank;
}

int MPIProxy::getSize() const{
	return _size;
}


} /* namespace utilities */
} /* namespace MPILib */
