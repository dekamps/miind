/*
 * ParallelException.cpp
 *
 *  Created on: 30.05.2012
 *      Author: david
 */

#include "ParallelException.hpp"
#include <boost/mpi/communicator.hpp>
#include <sstream>

namespace mpi = boost::mpi;

ParallelException::ParallelException(const char* message) :
		Exception(message) {
	mpi::communicator world;
	std::stringstream sstream;
	sstream << "Parallel Exception on processor: " << world.rank() << " from: "
			<< world.size() << " with error message: " << msg_;
	msg_ = sstream.str();
}

ParallelException::ParallelException(const std::string& message) :
		Exception(message) {
	mpi::communicator world;
	std::stringstream sstream;
	sstream << "Parallel Exception on processor: " << world.rank() << " from: "
			<< world.size() << " with error message: " << msg_;
	msg_ = sstream.str();
}

ParallelException::~ParallelException() throw () {
}

const char* ParallelException::what() const throw () {
	return msg_.c_str();
}
