// Copyright (c) 2005 - 2015 Marc de Kamps
//
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

#ifdef ENABLE_MPI

#ifndef  MPILIB_REPORT_HANDLER_ROOTCANVASMPI_HP
#define  MPILIB_REPORT_HANDLER_ROOTCANVASMPI_HP

#include <vector>
#include <memory>
#include <MPILib/include/TypeDefinitions.hpp>
#include "CanvasParameter.hpp"
#include "../ReportType.hpp"

class TGraph;


namespace MPILib {
/**
 * @brief Represents a ROOT canvas for the visualization of a running simulation. In general this class will not be created by the typical user.
 *
 * A RootReportHandler is responsible for logging the simulation results. When
 * ENABLE_MPI is not active, it is possible to add nodes to a canvas, and the
 * state and firing rate of each population is then rendered as the simulation
 * progresses. When ENABLE_MPI is not active all its functions are no-ops.
 */

class RootCanvas {
public:

	//! Set the dimensions of the canvas
	RootCanvas(const CanvasParameter&){}

	//! Copy constructor
	RootCanvas(const RootCanvas&){}

	//! Get the dimensions of the canvas
	CanvasParameter getCanvasParameter() const { return _par_canvas; }

	//! Command that will cause the rendering of TGraph* belonging to the NodeId
	void Render(report::ReportType, NodeId, TGraph*){}

	//! To be called by RootReportHandler
	void addNode(NodeId){}

private:
	CanvasParameter _par_canvas;
};

} //end of MPILib

#endif //ENABLE_MPI

#endif // include guard
