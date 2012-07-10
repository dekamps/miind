/*
 * MPIProxy.hpp
 *
 *  Created on: 10.07.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_MPIPROXY_HPP_
#define MPILIB_UTILITIES_MPIPROXY_HPP_
#include <MPILib/config.hpp>

#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
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

	template<typename T>
	void broadcast(T& value, int root);

private:
	static int _rank;

	static int _size;
};


template<typename T>
void MPIProxy::broadcast(T& value, int root){
#ifdef ENABLE_MPI
	mpi::communicator world;
	boost::mpi::broadcast(world, value, root);
#endif

}

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPIPROXY_HPP_ */
