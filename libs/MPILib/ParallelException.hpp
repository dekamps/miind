/*
 * ParallelException.hpp
 *
 *  Created on: 30.05.2012
 *      Author: david
 */

#ifndef PARALLELEXCEPTION_HPP_
#define PARALLELEXCEPTION_HPP_

#include "Exception.hpp"

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
	virtual ~ParallelException() throw();

    /** Returns a pointer to the constant error message.
     *  @return A pointer to a \c const \c char*. The underlying memory
     *          is in posession of the \c Exception object. Callers \a must
     *          not attempt to free the memory.
     */
    virtual const char* what() const throw ();
};

/** Convenience macros.
 */
#define miind_parallel_fail(ERROR_MESSAGE) throw (ParallelException(ERROR_MESSAGE))

#endif /* PARALLELEXCEPTION_HPP_ */
