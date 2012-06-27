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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef MPILIB_REPORT_REPORT_HPP_
#define MPILIB_REPORT_REPORT_HPP_

#include <MPILib/include/report/ReportType.hpp>
#include <MPILib/include/report/ReportValue.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>

#include <sstream>
#include <string>

namespace MPILib {
namespace report {

/**
 * @brief A Report is sent by a MPINode when it is queried.
 *
 * The information compiled by individual nodes is sent at a request of the agent that drives the
 * simulation, typically a MPINetwork. Such Reports are collected by an instance of an
 * AbstractReportHandler, which is responsible for entering this information into a simulation file.
 */
struct Report {
	/**
	 * Current time at this node.
	 */
	Time _time;
	/**
	 * Current firing rate of this node.
	 */
	Rate _rate;
	/**
	 * NodeId of this node.
	 */
	NodeId _id;
	/**
	 * The state space of the Algorithm
	 */
	algorithm::AlgorithmGrid _grid = (0);
	/**
	 * Whatever message should appear in the log file
	 */
	std::string _log_message;
	/**
	 * Information for the handler on how to treat the Report
	 */
	ReportType _type;
	/**
	 * Ad hoc values that need to be logged in the simulation file
	 */
	std::vector<ReportValue> _values;
	/**
	 * Number of nodes on the process, should be the same for all reports on one node
	 */
	unsigned int _nrNodes;

	/**
	 * Constructor
	 * @param time Current time at this node.
	 * @param rate Current firing rate of this node.
	 * @param id NodeId of this node.
	 * @param log_message Whatever message should appear in the log file
	 */
	Report(Time time, Rate rate, NodeId id, std::string log_message);

	/**
	 *
	 * @param time Current time at this node.
	 * @param rate Current firing rate of this node.
	 * @param id NodeId of this node.
	 * @param grid The state space of the Algorithm
	 * @param log_message Whatever message should appear in the log file
	 * @param type Information for the handler on how to treat the Report
	 * @param vec_values Ad hoc values that need to be logged in the simulation file
	 * @param nrNodes The number of nodes on this processor
	 */
	Report(Time time, Rate rate, NodeId id, algorithm::AlgorithmGrid grid,
			std::string log_message, ReportType type,
			std::vector<ReportValue> vec_values, int nrNodes = 0);

	/**
	 * Add a ReportValue to the report
	 * @param value The ReportValue added to the Report
	 */
	void addValue(const ReportValue& value);

};
// end of Report

}//end of namespace report
} // end of namespace MPILib

#endif // MPILIB_REPORT_REPORT_HPP_ include guard
