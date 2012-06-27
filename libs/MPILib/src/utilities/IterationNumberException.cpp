/*
 * IterationNumberException.cpp
 *
 *  Created on: 12.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/IterationNumberException.hpp>

namespace MPILib {
namespace utilities {

IterationNumberException::IterationNumberException(const char* message):Exception(message){}

IterationNumberException::IterationNumberException(const std::string& message):Exception(message){}


IterationNumberException::~IterationNumberException() throw(){
}

} /* namespace utilities */
} /* namespace MPILib */
