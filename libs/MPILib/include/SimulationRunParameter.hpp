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

#ifndef MPILIB_SIMULATIONRUNPARAMETER_HPP_
#define MPILIB_SIMULATIONRUNPARAMETER_HPP_

#include <string>
#include <MPILib/include/report/handler/AbstractReportHandler.hpp>
#include <MPILib/include/BasicTypes.hpp>



namespace MPILib
{
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


	class SimulationRunParameter
	{
	public:

		//! standard constructor
		SimulationRunParameter
		( 
			const report::handler::AbstractReportHandler&, 	/*!< ReportHandler (where and how is the NodeState information recorded ?)	*/
			Number,							/*!< maximum number of iterations											*/
			Time,   						/*!< Start time of simulation												*/
			Time,							/*!< End time of Simulation													*/
			Time,							/*!< Report time															*/ 
			Time,							/*!< Network step time														*/
			const std::string&,					/*!< Log file path name														*/
			Time report_state_time = 0		/*!< Report State time														*/
		); 

		//! copy constructor
		SimulationRunParameter
		(
			const SimulationRunParameter&
		);

		//! copy operator
		SimulationRunParameter&
				operator=(const SimulationRunParameter&);

		/// Give start time of simulation
		Time TBegin () const { return _t_begin;  }

		/// Give end time of simulation
		Time TEnd   () const { return _t_end;    }	

		/// Give report time of simulation
		Time TReport() const { return _t_report; }	

		Time TStep  () const { return _t_step;   }

		//! Give the time when a full state must be written out
		Time TState () const { return _t_state_report; }

		//! Give name of the log file, associated with this run
		std::string LogName() const { return _name_log; }

		const report::handler::AbstractReportHandler& Handler () const { return *_p_handler; }

		Number MaximumNumberIterations       () const { return _max_iter; }


	private:

		const report::handler::AbstractReportHandler*
				_p_handler;

		Number	_max_iter;

		Time	_t_begin;
		Time	_t_end;
		Time	_t_report;
		Time	_t_step;

		std::string	_name_log;

		Time	_t_state_report;

	}; // end of PopulationSimulationRunParameter


} // end of namespace MPILib

#endif // MPILIB_SIMULATIONRUNPARAMETER_HPP_ include guard
