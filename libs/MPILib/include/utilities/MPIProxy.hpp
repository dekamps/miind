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

/**
 * @brief A class to handle all MPI related code. It also provides works if MPI is disabled
 *
 * This class encapsulate all MPI related code. The class also works if MPI is not enabled,
 * such that the code does not depend if MPI is enabled or not. At the moment only in the
 * main method the MPI environment needs to be generated. All other MPI calls are handled
 * by this class
 */
class MPIProxy {
public:
	/**
	 * constructor sets the MPI rank and size
	 */
	MPIProxy();

	/**
	 * destructor
	 */
	virtual ~MPIProxy();

	/**
	 * wrapper method to return the process id, if mpi is disabled it returns 0
	 * @return the world rank of a process
	 */
	int getRank() const;

	/**
	 * wrapper method to return the size, if MPI is disabled it returns 1
	 * @return
	 */
	int getSize() const;

	/**
	 * wrapper for mpi barrier
	 */
	void barrier();

	/**
	 * waits until all request stored in the vector _mpiStatus are finished
	 */
	void waitAll();

	/**
	 * Broadcast the value from root
	 * @param value The value to be broadcast
	 * @param root The root process
	 */
	template<typename T>
	void broadcast(T& value, int root);

	/**
	 * asynchronous receive operation the mpi status is stored in _mpiStatus
	 * @param source The source of the message
	 * @param tag The tag of the message
	 * @param value The value received
	 */
	template<typename T>
	void irecv(int source, int tag, T& value) const;

	/**
	 * asynchronous send operation the mpi status is stored in _mpiStatus
	 * @param dest The destination of the message
	 * @param tag The tag of the message
	 * @param value The value sended
	 */
	template<typename T>
	void isend(int dest, int tag, const T& value) const;

private:

#ifdef ENABLE_MPI
	/**
	 * stores the mpi statuses
	 */
	static std::vector<boost::mpi::request> _mpiStatus;
#endif

	/**
	 * storage of the rank to avoid function calls
	 */
	static int _rank;

	/**
	 * storage of the size to avoid function calls
	 */
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
