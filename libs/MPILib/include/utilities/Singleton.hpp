// Copyright (c) 2005 - 2012 Marc de Kamps
// Copyright (c) 2011 - 2012 David-Matthias Sichau
// Copyright (c) 2010 Marc Kirchner
// Copyright (c) 2008 Thorben Kroeger
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

#ifndef MPILIB_UTILITIES_SINGLETON_HPP_
#define MPILIB_UTILITIES_SINGLETON_HPP_


#include <MPILib/include/utilities/Exception.hpp>
namespace MPILib {
namespace utilities {
/** Singleton holder template class.
 * Template class to create singletons. A singleton instance of class
 * MyType is created and accessed using
 * \code
 * typedef Singleton<MyType> MySingletonType;
 * MyType& myRef = MySingletonType::instance()
 * // ... do something ...
 * \endcode
 */
template<class T>
class Singleton
{
public:
    // disallow creation, copying and assignment

    /** Deleted constructor to disallow explicit construction.
     * Is not defined.
     */
    Singleton()=delete;

    /** Deleted copy constructor to disallow explicit copying.
     * Is not defined.
     * @param S A singleton object.
     */
    Singleton(const Singleton& S)=delete;

    /** Deleted assignment operator to disallow explicit assignment.
     * @param S A singleton object.
     * @return The current singleton.
     */
    Singleton& operator=(const Singleton& S)=delete;

    /** Return a reference to the only instance of \c Singleton<T>.
     * @return A reference to the instance of the object.
     */
    static T& instance();

    /** Destructor.
     */
    ~Singleton();

private:


    /** Create method. Creates the singleton instance (a Meyers singleton, ie.
     * a function static object) upon the first call to \c instance().
     */
    static void create();

    /** Pointer to the instance.
     */
    static T* pInstance_;

    /** Status of the singleton. True if the singleton was destroyed.
     */
    static bool destroyed_;

};

/** Returns the unique instance of class T. If it was already
 *  deleted an exception is thrown. If the class T was never used
 *  before a new instance is generated.
 *
 * @return Unique instance of class T
 */
template<class T> T& Singleton<T>::instance()
{
    if (!pInstance_) {
        if (destroyed_) {
            // dead reference
            throw Exception("The instance was already destroyed");
        } else {
            // initial creation
            create();
        }
    }
    return *pInstance_;
}

template<class T> Singleton<T>::~Singleton()
{
    pInstance_ = 0;
    destroyed_ = true;
}

template<class T> void Singleton<T>::create()
{
    static T theInstance;
    pInstance_ = &theInstance;
}

template<class T> T* Singleton<T>::pInstance_ = 0;
template<class T> bool Singleton<T>::destroyed_ = false;

}
}
#endif /* MPILIB_UTILITIES_SINGLETON_HPP_ */
