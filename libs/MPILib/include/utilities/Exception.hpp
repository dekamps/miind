// Copyright (c) 2005 - 2012 Marc de Kamps
//						2010 Marc Kirchner
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

#ifndef MPILIB_UTILITIES_EXCEPTION_HPP_
#define MPILIB_UTILITIES_EXCEPTION_HPP_

#include <string>
#include <exception>

namespace MPILib{
namespace utilities{

/** libpipe expection base class.
 */
class Exception : public std::exception
{
public:
    /** Constructor for C-style string error messages.
     *  @param message C-style string error message.
     *                 The string contents are copied upon construction
     *                 Responsibility for deleting the \c char* lies
     *                 with the caller. 
     */
    explicit Exception(const char* message);

    /** Constructor for STL string class error messages.
     *  @param message The error message.
     */
    explicit Exception(const std::string& message);

    /** Destructor. Nothrow guarantee.
     * Virtual to allow for subclassing.
     */
    virtual ~Exception() throw ();

    /** Returns a pointer to the constant error message.
     *  @return A pointer to a \c const \c char*. The underlying memory
     *          is in posession of the \c Exception object. Callers \a must
     *          not attempt to free the memory.
     */
    virtual const char* what() const throw ();

protected:
    /** Error message.
     */
    std::string msg_;
};

}//end namespace
}//end namespace


/** Convenience macros.
 */
#define miind_fail(ERROR_MESSAGE) throw (MPILib::utilities::Exception(ERROR_MESSAGE))


#endif //MPILIB_UTILITIES_EXCEPTION_HPP_
