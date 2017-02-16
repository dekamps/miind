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
#ifdef WANTROOT
#ifndef ENABLE_MPI

#ifndef  MPILIB_REPORT_HANDLER_ROOTCANVASNOMPI_HP
#define  MPILIB_REPORT_HANDLER_ROOTCANVASNOMPI_HP

#include <vector>
#include <memory>
#include <TCanvas.h>
#include <TGraph.h>
#include <TPad.h>
#include <TH2F.h>
#include <TStyle.h>
#include <MPILib/include/TypeDefinitions.hpp>
#include "CanvasParameter.hpp"
#include "../ReportType.hpp"



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
	RootCanvas(const CanvasParameter&);

	//! Copy constructor
	RootCanvas(const RootCanvas&);

	//! Get the dimensions of the canvas
	CanvasParameter getCanvasParameter() const;

	//! Command that will cause the rendering of TGraph* belonging to the NodeId
	void Render(report::ReportType, NodeId, TGraph*);

	//! To be called by RootReportHandler
	void addNode(NodeId);

private:

	void initializeCanvas();
	void SetMaximumDensity() const;
	void SetMaximumRate() const;

	void AddToCycle(NodeId);
	bool IsCycleComplete();

	int PadId(NodeId) const;

	const CanvasParameter _par_canvas;

	static std::unique_ptr<TCanvas> _p_canvas;

	static std::vector<NodeId>	_vec_id;
	static std::vector<int>	    _vec_scratch;

	static bool _b_rendering_started;

	static TH2F* _p_hist_rate;
	static TH2F* _p_hist_state;

	static TPad* _p_pad_rate;

	static TPad* _p_pad_state;

	static TStyle* _p_style;
};

} //end of MPILib

#endif //ENABLE_MPI

#endif /*  MPILIB_REPORT_HANDLER_INACTIVEREPORTHANDLER_HP */
#endif // don't bother if you don't want ROOT
