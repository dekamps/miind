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

#ifndef MPILIB_REPORT_HANDLER_ABSTRACTREPORTHANDLER_HPP_
#define MPILIB_REPORT_HANDLER_ABSTRACTREPORTHANDLER_HPP_

#include <string>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/report/Report.hpp>

namespace MPILib {
namespace report {
namespace handler {


/**
 * Base class for all ReportHandlers
 *
 * ReportHandlers are responsible for dispatching the Reports from each node and collating them
 * in a simulation results file. There are not many prescriptions for how this should be done and
 * it's very simple to derive one's own. RootReportHandler directly stores graphs of simulations.
 *
 */
class AbstractReportHandler {
public:

	/**
	 * Constructor
	 * @param fileName The filename of the output file @attention do not provide a extension this is
	 * done automatically
	 */
	AbstractReportHandler(const std::string& fileName);

	/**
	 * Manadatory virtual destructor
	 */
	virtual ~AbstractReportHandler();

	/**
	 * Writes the Report to the file.
	 * @param report The report written into the file
	 */
	virtual void writeReport(const Report& report) = 0;

	/**
	 * Cloning operation
	 * @return A clone of the ReportHandler
	 */
	virtual AbstractReportHandler* clone() const = 0;

	/**
	 * During Configuration a MPINode will associate itself with the handler.
	 * @param The NodeId of the Node
	 */
	virtual void initializeHandler(const NodeId&) = 0;

	/**
	 * A MPINode will request to be dissociated from the handler at the end of simulation.
	 * @param The NodeId of the Node
	 */
	virtual void detachHandler(const NodeId&) = 0;

	/**
	 * Getter for the actual output file, which is modified
	 * @return The name of the root output file
	 */
	std::string getRootOutputFileName() const;

	/**
	 * Getter for the unmodified stored string needed for copy constructor
	 * @return The unmodified fileName
	 */
	std::string getFileName() const;

private:

	/**
	 * The streamFileName without extension
	 */
	const std::string _streamFileName;

};

}// end namespace of handler
}// end namespace of report
}// end namespace of MPILib

#endif // MPILIB_REPORT_HANDLER_ABSTRACTREPORTHANDLER_HPP_ include guard
