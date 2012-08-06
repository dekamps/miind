// Copyright (c) 2005 - 2012 Marc de Kamps
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

#ifndef MPILIB_SIMULATIONRUNPARAMETER_HPP_
#define MPILIB_SIMULATIONRUNPARAMETER_HPP_

#include <string>
#include <MPILib/include/report/handler/AbstractReportHandler.hpp>
#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {
//! Parameter determining how a simulation is run. Specifiying begin and end time, log file names, etc.

//! This Parameter requires a handler, which determines how the simulation results are stored to disk. Common
//! choices are AsciiReportHandler or RootReportHandler (see their respective documentation). Begin and end times
//! of the simulation must be specified. Report time indicates at which time the simulation results should be written
//! by the Handler. Clearly one wants to set a report time that on the one hand represents the simulation accurately,
//! but on the other hand does not burden the simulation by writing out massive amounts of data, thereby impeding
//! simulation effciency. An update time allows to set a report time for an online visualisation module. The objective
//! of online visualization is typically to monitor whether the simulation behaves as expected, whilst running.
//! Since visualization can make a heavy demand on processing time on a single core machine (e.g. during development),
//! it makes sense to update much less than to report.
//! The maximum number of iterations is there to prevent endless loops or to specify a maximum number of
//! iterations that is reasonable. It allows for automatically breaking off simulations that have gone on expectedly
//! long. The string specfies the path of the log file, where the status of the simulation is reported during running.
//! In some simulations, typically involving population density techniques, the nodes have an activity as well as a
//! state. By default only activities are written into the simulation results by the handler, but optionally an
//! Algorithms state can be stored as well. This is specfied by the State Report time, which must be set in order
//! for more than just the beginning and the end state to be written out.
//! See the example programs in PopulistLib for applications.

class SimulationRunParameter {
public:

	/**
	 * The standard constructor
	 * @param handler ReportHandler (where and how is the NodeState information recorded ?)
	 * @param max_iter maximum number of iterations
	 * @param t_begin Start time of simulation
	 * @param t_end End time of Simulation
	 * @param t_report Report time
	 * @param t_step Network step time
	 * @param name_log Log file path name @attention without extension
	 * @param t_state_report Report State time
	 */
	SimulationRunParameter(
			const report::handler::AbstractReportHandler& handler,
			Number max_iter, Time t_begin, Time t_end, Time t_report,
			Time t_step, const std::string& name_log ="", Time t_state_report = 0);

	/**
	 * copy constructor
	 * @param parameter Another SimulationRunParameter
	 */
	SimulationRunParameter(const SimulationRunParameter& parameter);

	/**
	 * copy operator
	 * @param parameter Another SimulationRunParameter
	 */
	SimulationRunParameter&
	operator=(const SimulationRunParameter& parameter);

	/**
	 * Getter for the start time
	 * @return the start time
	 */
	Time getTBegin() const;

	/**
	 * Getter for the end time
	 * @return the end time
	 */
	Time getTEnd() const;

	/**
	 * Getter for the report time
	 * @return the report time
	 */
	Time getTReport() const;

	/**
	 * Getter for the step time
	 * @return the step time
	 */
	Time getTStep() const;

	/**
	 * Getter for the state report
	 * @return the state report time
	 */
	Time getTState() const;

	/**
	 * Getter for the log name
	 * @return the log file name
	 */
	std::string getLogName() const;

	/**
	 * Getter for the Handler
	 * @return the handler
	 */
	const report::handler::AbstractReportHandler& getHandler() const;

	/**
	 * Getter for the maximum number of iterations
	 * @return the maximum number of iterations
	 */
	Number getMaximumNumberIterations() const;

private:

	const report::handler::AbstractReportHandler* _pHandler;

	Number _maxIter;

	Time _tBegin;
	Time _tEnd;
	Time _tReport;
	Time _tStep;

	std::string _logFileName;

	Time _tStateReport;


};

} // end of namespace MPILib

#endif // MPILIB_SIMULATIONRUNPARAMETER_HPP_ include guard
