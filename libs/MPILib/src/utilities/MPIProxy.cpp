/*
 * MPIProxy.cpp
 *
 *  Created on: 10.07.2012
 *      Author: david
 */

#include <MPILib/config.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>
#include <MPILib/include/utilities/Exception.hpp>




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

void MPIProxy::barrier(){
#ifdef ENABLE_MPI
	mpi::communicator world;
	world.barrier();
#endif
}

void MPIProxy::waitAll(){
#ifdef ENABLE_MPI
	mpi::wait_all(_mpiStatus.begin(), _mpiStatus.end());
	_mpiStatus.clear();
#endif
}

#ifdef ENABLE_MPI
std::vector<boost::mpi::request> MPIProxy::_mpiStatus;
#endif


} /* namespace utilities */
} /* namespace MPILib */
