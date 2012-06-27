// Copyright (c) 2005 - 2011 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef MPILIB_REPORT_HANDLER_GRAPHKEY_HPP_
#define MPILIB_REPORT_HANDLER_GRAPHKEY_HPP_

#include <string>
#include <MPILib/include/BasicTypes.hpp>

namespace MPILib {
namespace report {
namespace handler {

enum GraphType {
	STATEGRAPH, RATEGRAPH
};

/**
 * @brief Serves to interpret the name of a graph assigned by any AbstractReportHandler, and serves as a
 * key for searches on graphs in simulation files.
 *
 * Given the size of root files nowadays, the Node and time stamp of state graphs in any handler must not
 * only be stored, but also be retrieved in subsequent analysis. The GraphKey object is the central object
 * to code and decode the names of state graphs.
 */
struct GraphKey {

	/**
	 * The nodeId of the node
	 */
	NodeId _id = NodeId(0);
	/**
	 * The time point of the node
	 */
	Time _time = 0.0;
	/**
	 * The GraphType
	 */
	GraphType _type = RATEGRAPH;

	/**
	 * Default constructor for use in containers
	 */
	GraphKey();

	/**
	 * construct a graph key from the key in the root file.
	 * If the string does not represent a valid key, no object will
	 * be constructed, but otherwise nothing will happen.
	 * This allows parsing of heterogeneous object files.
	 * @param key_string The graph key
	 */
	GraphKey(const std::string& key_string);

	/**
	 * construct a graph key from a Report information
	 * @param id The NodeId of the Node
	 * @param time The Time point
	 */
	GraphKey(NodeId id, Time time);

	/**
	 * generates a name of the graph
	 * @return A name for the graph
	 */
	std::string generateName() const;
};

} // end namespace of handler
} // end namespace of report
} // end namespace of MPILib
#endif // include guard
