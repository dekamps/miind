// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef MPILIB_UTILITIES_MPIPROXY_HPP_
#define MPILIB_UTILITIES_MPIPROXY_HPP_
//#include <MPILib/config.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#include <MPILib/include/utilities/Singleton.hpp>
#include <MPILib/include/utilities/Log.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/request.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/nonblocking.hpp>
namespace mpi = boost::mpi;
#endif // ENABLE_MPI

namespace MPILib {
namespace utilities {


/**
 * @brief A class to handle all MPI related code. It also provides works if MPI is disabled
 *
 * MIIND relies on BOOST.MPI to simplify MPI calling.This class encapsulate all MPI related code. The class also works if MPI is not enabled,
 * so that the code does not depend on whether MPI is enabled or not. At the moment only in the
 * main method the MPI environment needs to be generated, i.e. the main program requires an
 * mpi::environment instance to communicate run parameters, such as the number of processors
 * to the program. Consult the largeNetwork program for an example on its definition.
 * All other MPI calls are handled
 * by this class, which encapsulates the mpi::world object. MIIND uses broadcasts and blocking  irecv and isend calls for point-to-point calls.
 * 
 * 
 */
class MPIProxy_ {
public:
	/**
	 * destructor
	 */
	virtual ~MPIProxy_();

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
	 * waits until all requests stored in the vector _mpiStatus are finished
	 */
	void waitAll();

	/**
	 * Broadcast the value from root
	 * @param value The value to be broadcast
	 * @param root The root process
	 */
	template<typename T>
	void broadcast(T&, int);

	/**
	 * asynchronous receive operation the mpi status is stored in _mpiStatus
	 * @param source The source of the message
	 * @param tag The tag of the message
	 * @param value The value received
	 */
	template<typename T>
	void irecv(int, int, T&) const;

	/**
	 * asynchronous send operation the mpi status is stored in _mpiStatus
	 * @param dest The destination of the message
	 * @param tag The tag of the message
	 * @param value The value sended
	 */
	template<typename T>
	void isend(int, int, const T&) const;


private:
	/**
	 * Declare the Singleton class a friend to allow construction of the MPIProxy_ class
	 */
	friend class Singleton<MPIProxy_>;
	/**
	 * constructor sets the MPI rank and size
	 */
	MPIProxy_();

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
void MPIProxy_::broadcast(T& value, int root) {
#ifdef ENABLE_MPI
	mpi::communicator world;
	boost::mpi::broadcast(world, value, root);
#endif
}

template<typename T>
void MPIProxy_::irecv(int source, int tag, T& value) const {
#ifdef ENABLE_MPI
	mpi::communicator world;
	_mpiStatus.push_back(world.irecv(source, tag, value));
	LOG(utilities::logDEBUG4)<<"recv source: "<<source<<"; tag: "<<tag<<"; value: "<<value;
#else
	MPILib::utilities::Exception("MPI Code called from serial code in irecv");
#endif
}

template<typename T>
void MPIProxy_::isend(int dest, int tag, const T& value) const {
#ifdef ENABLE_MPI
	mpi::communicator world;
	_mpiStatus.push_back(world.isend(dest, tag, value));
	LOG(utilities::logDEBUG4)<<"send destination: "<<dest<<"; tag: "<<tag<<"; value: "<<value;
#else
	MPILib::utilities::Exception("MPI Code called from serial code in isend");
#endif
}

/**
 * Generate an singleton instance of the MPIProxy_ class.
 */
typedef Singleton<MPIProxy_> MPIProxySingleton;

/**
 * Wrapper function to reduce the writing needed to access the MPIProxy_.
 * inline this function to allow to definition in multiple translation units.
 * @return The reference to the single instance of MPIProxy
 */
inline MPIProxy_& MPIProxy() {
	return MPIProxySingleton::instance();
}



} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_MPIPROXY_HPP_ */
