/*
 * TimeException.hpp
 *
 *  Created on: 12.06.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_TIMEEXCEPTION_HPP_
#define MPILIB_UTILITIES_TIMEEXCEPTION_HPP_

#include <MPILib/include/utilities/Exception.hpp>

namespace MPILib {
namespace utilities {

class TimeException: public Exception {
public:

    /** Constructor for C-style string error messages.
     *  @param message C-style string error message.
     *                 The string contents are copied upon construction
     *                 Responsibility for deleting the \c char* lies
     *                 with the caller.
     */
    explicit TimeException(const char* message);

    /** Constructor for STL string class error messages.
     *  @param message The error message.
     */
    explicit TimeException(const std::string& message);

	virtual ~TimeException() throw();
};

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_TIMEEXCEPTION_HPP_ */
