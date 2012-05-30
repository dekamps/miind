/*
 * ParallelException.cpp
 *
 *  Created on: 30.05.2012
 *      Author: david
 */

#include "ParallelException.hpp"
#include <boost/mpi/communicator.hpp>

namespace mpi = boost::mpi;

ParallelException::ParallelException(const char* message) :
    msg_(message)
{
	mpi::communicator world;
	msg_<<"\t processor rank: "<<world.rank()<<"\t from: "<<world.size();
}

ParallelException::ParallelException(const std::string& message) :
    msg_(message)
{
	mpi::communicator world;
	msg_<<"\t processor rank: "<<world.rank()<<"\t from: "<<world.size();
}

ParallelException::~ParallelException() throw ()
{
}

const char* ParallelException::what() const throw ()
{
    return msg_.c_str();
}
