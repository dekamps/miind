/*
 * ParallelException.hpp
 *
 *  Created on: 30.05.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_PARALLELEXCEPTION_HPP_
#define MPILIB_UTILITIES_PARALLELEXCEPTION_HPP_

#include <MPILib/include/utilities/Exception.hpp>

namespace MPILib {
namespace utilities {

class ParallelException: public Exception {
public:
	/** Constructor for C-style string error messages.
	 *  @param message C-style string error message.
	 *                 The string contents are copied upon construction
	 *                 Responsibility for deleting the \c char* lies
	 *                 with the caller.
	 */
	explicit ParallelException(const char* message);

	/** Constructor for STL string class error messages.
	 *  @param message The error message.
	 */
	explicit ParallelException(const std::string& message);

	/** Destructor. Nothrow guarantee.
	 * Virtual to allow for subclassing.
	 */
	virtual ~ParallelException() throw ();

	/** Returns a pointer to the constant error message.
	 *  @return A pointer to a \c const \c char*. The underlying memory
	 *          is in posession of the \c Exception object. Callers \a must
	 *          not attempt to free the memory.
	 */
	virtual const char* what() const throw ();
};

} //end namespace
} //end namespace

/** Convenience macros.
 */
#define miind_parallel_fail(ERROR_MESSAGE) throw (MPILib::utilities::ParallelException(ERROR_MESSAGE))

#endif /* MPILIB_UTILITIES_PARALLELEXCEPTION_HPP_ */
