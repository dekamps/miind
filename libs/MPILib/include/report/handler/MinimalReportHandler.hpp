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


#ifndef MPILIB_REPORT_HANDLER_MINIMALREPORTHANDLER_HPP_
#define MPILIB_REPORT_HANDLER_MINIMALREPORTHANDLER_HPP_

#include <memory>
#include <fstream>
#include "AbstractReportHandler.hpp"
#include "CanvasParameter.hpp"

namespace MPILib {
namespace report {
namespace handler {

	class MinimalReportHandler : public AbstractReportHandler {
	public:

		MinimalReportHandler
		(
			const std::string&,  		//!< results file name
			bool b_force_state = false,	//!< if true the algorithm is prompted to write out its state (but the algorithm may or may not take this hint; check the algorithm docu)
			bool b_canvas = false,	    //!< ineffective; place holder variable
			const CanvasParameter& = DEFAULT_CANVAS
		);

		virtual ~MinimalReportHandler();


		MinimalReportHandler(const MinimalReportHandler&);


		/**
		 * Writes the Report to the file.
		 * @param report The report written into the file
		 */
		virtual void writeReport(const Report& report);


		/**
		 * Cloning operation
		 * @return A clone of the ReportHandler
		 */
		virtual MinimalReportHandler* clone() const;

		/**
		 * During Configuration an MPINode will associate itself with the handler.
		 * @param nodeId The NodeId of the Node
		 */
		virtual void initializeHandler(const NodeId& nodeId);

		/**
		 * A MPINode will request to be dissociated from the handler at the end of simulation.
		 * @param nodeId The NodeId of the Node
		 */
		virtual void detachHandler(const NodeId& nodeId);

		/**
		 *  A no-op for this handler
		 */
		void addNodeToCanvas(NodeId){}



	private:

		static std::ofstream _ofst;
	};

} // end of MPILib
} // end of report
} // end of handler


#endif /* MINIMALREPORTHANDLER_HPP_ */
