/*
 * MPIProxy.hpp
 *
 *  Created on: 10.07.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_MPIPROXY_HPP_
#define MPILIB_UTILITIES_MPIPROXY_HPP_
#include <MPILib/config.hpp>



namespace MPILib {
namespace utilities {

class MPIProxy {
public:
	MPIProxy();
	virtual ~MPIProxy();

	int getRank() const;

	int getSize() const;

private:
	static int _rank;

	static int _size;
};

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPIPROXY_HPP_ */
