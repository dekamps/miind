// Copyright (c) 2005 - 2014 Marc de Kamps
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
#ifndef MPILIB_CUSTOMCONNECTIONPARAMETERS_HPP_
#define MPILIB_CUSTOMCONNECTIONPARAMETERS_HPP_

#include <limits>
#include <map>
#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {

/**
 * Connection parameters for a network of OrnsteinUhlenbeck populations
 * need two parameters: a number parameter, which is not squared in the
 * calculation of sigma, and an efficacy, which is squared.
 */
class CustomConnectionParameters {
	public:
  std::map<std::string, std::string> _params;

	/**
	 * default constructor
	 */
	CustomConnectionParameters()=default;

  void setParam(std::string k, std::string v) {
    _params[k] = v;
  }

	/**
	 * construct, using effective number of connections and effective efficacy
	 * @param number_of_connections effective number of connections
	 * @param efficacy synaptic efficacy
	 * @param delay  delay of connection
	 */
	CustomConnectionParameters(std::map<std::string, std::string> params
	) :
	_params(params) {}
};

/**
 * Necessary for tests on Dale's law
 * @param connection The connection
 * @return The efficacy of the connection
 */
inline std::string getParam(const CustomConnectionParameters& connection, const std::string p_name) {
	return connection._params.at(p_name);
}

/**
 * Necessary for tests on Dale's law
 * @param connection The connection
 * @return The efficacy of the connection
 */
inline double toEfficacy(const CustomConnectionParameters& connection) {
	return std::stod(connection._params.at("efficacy"));
}

}

#endif
