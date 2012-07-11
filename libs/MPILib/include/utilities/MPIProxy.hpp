/*
 * MPIProxy.hpp
 *
 *  Created on: 10.07.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_MPIPROXY_HPP_
#define MPILIB_UTILITIES_MPIPROXY_HPP_
#include <MPILib/config.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/request.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/nonblocking.hpp>
namespace mpi = boost::mpi;
#endif

namespace MPILib {
namespace utilities {

class MPIProxy {
public:
	MPIProxy();
	virtual ~MPIProxy();

	int getRank() const;

	int getSize() const;

	void barrier();

	template<typename T>
	void broadcast(T& value, int root);

	void waitAll();

	template<typename T>
	void irecv(int source, int tag, T& value) const;

	template<typename T>
	void isend(int dest, int tag, const T& value) const;

private:

#ifdef ENABLE_MPI
	static std::vector<boost::mpi::request> _mpiStatus;
#endif

	static int _rank;

	static int _size;
};

template<typename T>
void MPIProxy::broadcast(T& value, int root) {
#ifdef ENABLE_MPI
	mpi::communicator world;
	boost::mpi::broadcast(world, value, root);
#endif
}

template<typename T>
void MPIProxy::irecv(int source, int tag, T& value) const {
#ifdef ENABLE_MPI
	mpi::communicator world;
	_mpiStatus.push_back(world.irecv(source, tag, value));
#else
	MPILib::utilities::Exception("MPI Code called from serial code in irecv");
#endif
}

template<typename T>
void MPIProxy::isend(int dest, int tag, const T& value) const {
#ifdef ENABLE_MPI
	mpi::communicator world;
	_mpiStatus.push_back(world.isend(dest, tag, value));
#else
	MPILib::utilities::Exception("MPI Code called from serial code in isend");
#endif
}

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPIPROXY_HPP_ */
