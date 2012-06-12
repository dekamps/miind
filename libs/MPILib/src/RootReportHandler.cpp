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

#ifdef WIN32
// All these warnings relate to ROOT source code
#pragma warning(disable: 4267 4305 4800 4996)
#endif 

#include <algorithm>
#include <sstream>
#include <assert.h>
#include <MPILib/include/RootReportHandler.hpp>
#include <MPILib/include/utilities/Exception.hpp>

#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/GraphKey.hpp>

#ifdef HAVE_LIBCORE
#include <TApplication.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TNtuple.h>
#include <TStyle.h>

using namespace ROOT;

#endif // end of ROOT includes
#ifndef HAVE_LIBCORE // Without ROOT, a NO-OP
#else  //ROOT defined !!

namespace MPILib {

const int NUMBER_HISTO_CHANNELS = 500;

const Time T_MIN = 0;
const Time T_MAX = .04;
const Density DENSITY_MIN = -0.1;
const Density DENSITY_MAX = 50;
const Rate F_MIN = -1;
const Rate F_MAX = 20;
const Potential POTENTIAL_MIN = -0.01;
const Potential POTENTIAL_MAX = 0.020;

namespace {

// need one global application object
TApplication APPLICATION("application", 0, 0);
TH2F HISTO_RATE("rate_histo", "", NUMBER_HISTO_CHANNELS, T_MIN, T_MAX,
		NUMBER_HISTO_CHANNELS, F_MIN, F_MAX);

TH2F HISTO_STATE("state_histo", "", NUMBER_HISTO_CHANNELS, POTENTIAL_MIN,
		POTENTIAL_MAX, NUMBER_HISTO_CHANNELS, DENSITY_MIN, DENSITY_MAX);
}

TFile* RootReportHandler::_p_file = 0;
TCanvas* RootReportHandler::_p_canvas = 0;
TNtuple* RootReportHandler::_p_tuple = 0;
TPad* RootReportHandler::_p_pad_rate = 0;
TPad* RootReportHandler::_p_pad_state = 0;

std::vector<NodeId> RootReportHandler::_list_nodes(0);
std::vector<NodeId> RootReportHandler::_vector_id(0);

ValueHandlerHandler RootReportHandler::_value_handler;

RootReportHandler::RootReportHandler(const string& file_name, bool b_canvas,
		bool b_file, const CanvasParameter& par) :
		AbstractReportHandler(file_name), //
		_p_style(new TStyle), //
		_p_current_rate_graph(0), //
		_p_current_state_graph(0), //
		_b_canvas(b_canvas), //
		_b_file(b_file), //
		_nr_reports(0), //
		_index_pad(-1), //
		_par_canvas(par) {
	_p_style->SetOptStat(0);
}

RootReportHandler::RootReportHandler(const RootReportHandler& rhs) :
		AbstractReportHandler(rhs.MediumName()), _p_current_rate_graph(0), //
		_p_current_state_graph(0), //
		_b_canvas(rhs._b_canvas), //
		_b_file(rhs._b_file), //
		_nr_reports(0), //
		_index_pad(-1), //
		_par_canvas(rhs._par_canvas) {
	if (rhs._p_current_rate_graph)
		throw utilities::Exception(STR_HANDLER_STALE);
}

RootReportHandler::~RootReportHandler()
// 11-07-2007: test on p_tuple by Volker Baier
{
	if (_p_file) {
		if (_p_tuple)
			_p_tuple->Write();
		_p_file->Close();
		delete _p_file;
		_p_file = 0;
	}

	if (_p_canvas) {
		delete _p_canvas;
		_p_canvas = 0;
	}
}

bool RootReportHandler::WriteReport(const Report& report) {

	if (_nr_reports == 0) {
		if (_par_canvas._t_max != 0)
			this->SetTimeRange(0, _par_canvas._t_max);

		_p_current_rate_graph = new TGraph;

		std::ostringstream stream;
		stream << "rate_" << report._id;
		_p_current_rate_graph->SetName(stream.str().c_str());

		vector<NodeId>::iterator iter = find(_vector_id.begin(),
				_vector_id.end(), report._id);

		if (iter != _vector_id.end())
			_index_pad = static_cast<int>(iter - _vector_id.begin()) + 1;
		// else
		// ignore
	}

	_p_current_rate_graph->SetPoint(_nr_reports++, report._time, report._rate);

	delete _p_current_state_graph;
	_p_current_state_graph = ConvertAlgorithmGridToGraph(report);

	if (report._type == STATE && BelongsToAnAlgorithm()
			&& (IsStateWriteMandatory()))
		_p_current_state_graph->Write();

	// always log ReportValue elements
	_value_handler.AddReport(report);
	return true;
}

RootReportHandler* RootReportHandler::Clone() const {
	// Cloning happens at configure time.
	// Now is the time to divide the pads

	if (_p_canvas && !_p_pad_rate)
		InitializeCanvas();

	return new RootReportHandler(*this);
}

TGraph* RootReportHandler::ConvertAlgorithmGridToGraph(
		const Report& report) const {

	vector<double> vector_of_grid_values = report._grid.ToStateVector();

	// if the Report does not contain a filled AlgorithmGrid, no Graph can be made
	if (vector_of_grid_values.size() == 0)
		return 0;

	vector<double> vector_of_state_interpretation =
			report._grid.ToInterpretationVector();

	TGraph* p_state_graph = new TGraph;

	GraphKey key(report._id, report._time);
	p_state_graph->SetName(key.Name().c_str());

	assert(
			vector_of_grid_values.size() == vector_of_state_interpretation.size());
	for (std::vector<double>::iterator iter = vector_of_grid_values.begin();
			iter != vector_of_grid_values.end(); iter++) {
		int n_index = static_cast<int>(iter - vector_of_grid_values.begin());

		p_state_graph->SetPoint(n_index,
				vector_of_state_interpretation[n_index],
				vector_of_grid_values[n_index]);

	}

	return p_state_graph;
}

bool RootReportHandler::BelongsToAnAlgorithm() const {
	return (_p_current_rate_graph != 0);
}

bool RootReportHandler::IsStateWriteMandatory() const {
	return _b_file;
}

bool RootReportHandler::HasANoneTrivialState(const Report& report) const {
	return true;
}

void RootReportHandler::AddNodeToCanvas(NodeId id) const {
	_vector_id.push_back(id);
}

void RootReportHandler::InitializeCanvas() const {
	this->SetMaximumDensity();
	this->SetMaximumRate();

	_p_pad_rate = new TPad("rate", "", 0.05, 0.05, 0.45, 0.95);
	_p_pad_state = new TPad("state", "", 0.55, 0.05, 0.95, 0.95);

	int number_of_drawable_populations = _vector_id.size();

	_p_pad_rate->Draw();
	_p_pad_state->Draw();
	_p_pad_rate->cd();
	_p_pad_rate->Divide(1, number_of_drawable_populations, 1e-5, 1e-5);

	_p_pad_state->cd();
	_p_pad_state->Divide(1, number_of_drawable_populations, 1e-5, 1e-5);

	_p_canvas->cd();
	_p_canvas->Update();
}

void RootReportHandler::ToRateCanvas(int n_index) {

}

void RootReportHandler::InitializeHandler(const NodeId& info) {
	// Purpose: this function will be called by DynamicNode upon configuration. This means that if there is no canvas yet, and it is desired,
	// that now is the time to create it. This works under the assumption that no two DynamicNetwork simulations are running at the same time.
	// Idea is that several handlers can be created, but that no competition between the ROOT resources, which contain globals and statics, takes place
	// Assumptions: No two DynamicNetwork simulations are running at the same time
	// Author: Marc de Kamps
	// Date: 26-08-2005

	if (!_p_file) {
		_p_file = new TFile(this->MediumName().c_str(), "RECREATE");

		if (_p_file->IsZombie())
			throw utilities::Exception(STR_ROOT_FILE_OPENED_FAILED);

		_p_tuple = new TNtuple("infotuple", "node info", "id:x:y:z:f");
		_value_handler.Reset();

	}

	WriteInfoTuple(info);

	if (_b_canvas) {

		if (!_p_canvas) {
			_p_canvas = new TCanvas(CANVAS_NAME.c_str(), CANVAS_TITLE.c_str(),
					CANVAS_X_DIMENSION, CANVAS_Y_DIMENSION);
			InitializeCanvas();
		}
	}
}

void RootReportHandler::WriteInfoTuple(const NodeId& nodeId) {
	_p_tuple->Fill(static_cast<Float_t>(nodeId));
	_list_nodes.push_back(nodeId);
}

void RootReportHandler::DetachHandler(const NodeId& nodeId) {
	// Purpose: this function will be called upon DynamicNode destruction. 
	// This works under the assumption that no isolated DynamicNodes
	// exist which are associated with an open RootReporthandler. 
	// Author: Marc de Kamps
	// Date: 26-08-2005

	RemoveFromNodeList(nodeId);

	if (_p_current_rate_graph) {
		_p_current_rate_graph->Write();
		delete _p_current_rate_graph;
		_p_current_rate_graph = 0;
		if (!_value_handler.IsWritten())
			_value_handler.Write();

	}

	if (_list_nodes.empty())
		GlobalCleanUp();
}

void RootReportHandler::SetMaximumDensity() const {
	if (_par_canvas._dense_max > 0) {
		int n_x = HISTO_STATE.GetXaxis()->GetNbins();
		int n_y = HISTO_STATE.GetYaxis()->GetNbins();
		double y_min = HISTO_STATE.GetYaxis()->GetXmin();

		HISTO_STATE.SetBins(n_x, _par_canvas._state_min, _par_canvas._state_max,
				n_y, y_min, _par_canvas._dense_max);

	}
}

void RootReportHandler::SetMaximumRate() const {
	if (_par_canvas._f_max > 0) {
		int n_x = HISTO_RATE.GetXaxis()->GetNbins();
		int n_y = HISTO_RATE.GetYaxis()->GetNbins();
		double y_min = HISTO_RATE.GetYaxis()->GetXmin();

		HISTO_RATE.SetBins(n_x, _par_canvas._t_min, _par_canvas._t_max, n_y,
				y_min, _par_canvas._f_max);
	}
}

void RootReportHandler::RemoveFromNodeList(NodeId id) {
	vector<NodeId>::iterator iter = find(_list_nodes.begin(), _list_nodes.end(),
			id);

	if (iter == _list_nodes.end())
		throw utilities::Exception(
				"Can't locate NodeId during detaching handler");

	_list_nodes.erase(iter);
}

void RootReportHandler::GlobalCleanUp() {
	_p_tuple->Write();
	_p_file->Close();

	delete _p_pad_state;
	_p_pad_state = 0;

	delete _p_pad_rate;
	_p_pad_rate = 0;

	delete _p_canvas;
	_p_canvas = 0;

	delete _p_file;
	_p_file = 0;

	_p_tuple = 0;
	_list_nodes.clear();
	_vector_id.clear();
}

void RootReportHandler::SetDensityRange(Density d_min,
		Density d_max) {
	int n_x = HISTO_STATE.GetXaxis()->GetNbins();
	int n_y = HISTO_STATE.GetYaxis()->GetNbins();

	_par_canvas._dense_min = d_min;
	_par_canvas._dense_max = d_max;

	HISTO_STATE.SetBins(n_x, _par_canvas._state_min, _par_canvas._state_max,
			n_y, _par_canvas._dense_min, _par_canvas._dense_max);
}

void RootReportHandler::SetFrequencyRange(Rate r_min, Rate r_max) {

	int n_x = HISTO_RATE.GetXaxis()->GetNbins();
	int n_y = HISTO_RATE.GetYaxis()->GetNbins();

	_par_canvas._f_min = r_min;
	_par_canvas._f_max = r_max;
	HISTO_RATE.SetBins(n_x, _par_canvas._state_min, _par_canvas._state_max, n_y,
			_par_canvas._f_min, _par_canvas._f_max);
}

void RootReportHandler::SetTimeRange(Time t_min,
		Time t_max) {
	int n_x = HISTO_RATE.GetXaxis()->GetNbins();
	int n_y = HISTO_RATE.GetYaxis()->GetNbins();

	_par_canvas._t_min = t_min;
	_par_canvas._t_max = t_max;

	HISTO_RATE.SetBins(n_x, _par_canvas._t_min, _par_canvas._t_max, n_y,
			_par_canvas._f_min, _par_canvas._f_max);

}

void RootReportHandler::SetPotentialRange(Potential v_min, Potential v_max) {
	int n_x = HISTO_STATE.GetXaxis()->GetNbins();
	int n_y = HISTO_STATE.GetYaxis()->GetNbins();

	_par_canvas._state_min = v_min;
	_par_canvas._state_max = v_max;

	HISTO_STATE.SetBins(n_x, _par_canvas._state_min, _par_canvas._state_max,
			n_y, _par_canvas._dense_min, _par_canvas._dense_max);
}

} //end namespace

#endif

