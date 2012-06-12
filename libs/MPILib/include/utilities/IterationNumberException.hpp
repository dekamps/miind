/*
 * IterationNumberException.hpp
 *
 *  Created on: 12.06.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_ITERATIONNUMBEREXCEPTION_HPP_
#define MPILIB_UTILITIES_ITERATIONNUMBEREXCEPTION_HPP_

#include <MPILib/include/utilities/Exception.hpp>

namespace MPILib {
namespace utilities {

class IterationNumberException: public Exception {
public:

    /** Constructor for C-style string error messages.
     *  @param message C-style string error message.
     *                 The string contents are copied upon construction
     *                 Responsibility for deleting the \c char* lies
     *                 with the caller.
     */
    explicit IterationNumberException(const char* message);

    /** Constructor for STL string class error messages.
     *  @param message The error message.
     */
    explicit IterationNumberException(const std::string& message);

	virtual ~IterationNumberException() throw();
};

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_ITERATIONNUMBEREXCEPTION_HPP_ */
