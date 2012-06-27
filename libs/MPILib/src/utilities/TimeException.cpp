/*
 * IterationNumberException.cpp
 *
 *  Created on: 12.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/TimeException.hpp>

namespace MPILib {
namespace utilities {

TimeException::TimeException(const char* message):Exception(message){}

TimeException::TimeException(const std::string& message):Exception(message){}


TimeException::~TimeException() throw(){
}

} /* namespace utilities */
} /* namespace MPILib */
