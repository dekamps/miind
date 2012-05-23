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

#ifndef _CODE_LIBS_DYNAMICLIB_ROOTHANDLER_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_ROOTHANDLER_INCLUDE_GUARD

#include <string>
#include <vector>
#include "AbstractReportHandler.h"
#include "BasicDefinitions.h"
#include "CanvasParameter.h"
#include "ValueHandlerHandler.h"

using std::vector;
using std::string;

// forward declarations 

class TApplication;
class TPad;
class TGraph;
class TCanvas;
class TFile;
class TH2F;
class TNtuple;
class TStyle;


namespace DynamicLib
{
	//! This handler writes states and firing rates as TGraph objects in a root file.
	//! (see http://root.cern.ch). It is also able to show run time graphs of selected nodes.
	//!
	//! ROOT is a visualisation and data management tool with a good interface to numerical
	//! methods. The RootReportHandler is reliable when no run time graphs are shown and is a very
	//! efficient way to store the simulation data, as they are compressed when written to file.
	//! The behaviour and resource consumption of the run time graphs need closer investigation
	//! but for debugging purposes they are tremendously useful. A disadvantage for high throughput is that
	//! memory use increases over time. Where this is an issue, use RootHighThroughputHandler.
	class RootReportHandler : public AbstractReportHandler
	{
	public:

		RootReportHandler
			(
				const  string&,
				bool   b_canvas             = false,
				bool   b_force_state_write  = false,
				const CanvasParameter&	    = DEFAULT_CANVAS
			);

		RootReportHandler(const RootReportHandler&);

		//! virtual destructor
		virtual ~RootReportHandler();

		//! Collects the Report of a DynamicNode for storage in the simulation file.
		virtual bool WriteReport(const Report&);

		//! Triggers updating of the run time graphics. Is called by the DynamicNetworkImplementation
		virtual bool Update();

		virtual RootReportHandler* Clone() const;

		virtual void InitializeHandler
		(
				const NodeInfo&
		);


		virtual void DetachHandler
			(
				const NodeInfo&
			);

		virtual void AddNodeToCanvas(NodeId) const;

		//! Set the minimum and maximum density to be shown in the canvas.
		void SetDensityRange
		(
			Density,
			Density
		);

		void SetFrequencyRange
		(
			Rate,
			Rate
		);

		void SetTimeRange
		(
			Time,
			Time
		);

		void SetPotentialRange
		(
			Potential,
			Potential
		);

	private:

		void InitializeCanvas() const;
		void ToRateCanvas(int);
		void SetMaximumRate() const;
		void SetMaximumDensity() const;

		void WriteInfoTuple(const NodeInfo&);
		void RemoveFromNodeList(NodeId);
		void GlobalCleanUp();

		TGraph* ConvertAlgorithmGridToGraph(const Report&) const;
		bool BelongsToAnAlgorithm  () const;
		bool IsStateWriteMandatory () const;
		bool HasANoneTrivialState  (const Report&) const;

		bool HandleReportValue(const Report&);


		static TCanvas*			_p_canvas;
		static TFile*			_p_file;
		static TNtuple*			_p_tuple;
		static TPad*			_p_pad_rate;
		static TPad*			_p_pad_state;
		TStyle*					_p_style;

		static ValueHandlerHandler	_value_handler;

	
		static vector<NodeId>	_list_nodes;
		static vector<NodeId>	_vector_id;

		TGraph*					_p_current_rate_graph;
		TGraph*					_p_current_state_graph;

		bool					_b_canvas;
		bool					_b_file;

		int						_nr_reports;
		int						_index_pad;

		CanvasParameter		_par_canvas;

	};

} // end of DynamicLib

#endif // include guard
