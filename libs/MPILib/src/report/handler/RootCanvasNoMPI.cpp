
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
#ifndef ENABLE_MPI

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <MPILib/include/report/handler/RootCanvasNoMPI.hpp>
#include <TCanvas.h>
#include <TH2F.h>
#include <TApplication.h>
#include <TStyle.h>

const int CANVAS_X_DIMENSION    = 800;
const int CANVAS_Y_DIMENSION    = 800;
const int NUMBER_HISTO_CHANNELS = 500;

// need one global application object
TApplication APPLICATION("application",0,0);
TStyle* MPILib::RootCanvas::_p_style  = new TStyle;

std::unique_ptr<TCanvas> MPILib::RootCanvas::_p_canvas(nullptr);
std::vector<MPILib::NodeId> MPILib::RootCanvas::_vec_id(0);

std::vector<int> MPILib::RootCanvas::_vec_scratch(0);
bool MPILib::RootCanvas::_b_rendering_started = false;
TH2F* MPILib::RootCanvas::_p_hist_state = 0;
TH2F* MPILib::RootCanvas::_p_hist_rate = 0;

TPad* MPILib::RootCanvas::_p_pad_state = 0;
TPad* MPILib::RootCanvas::_p_pad_rate  = 0;

using namespace MPILib;

RootCanvas::RootCanvas(const CanvasParameter& par_canvas):
_par_canvas(par_canvas)
{
}

CanvasParameter RootCanvas::getCanvasParameter() const
{
	return _par_canvas;
}

int RootCanvas::PadId(NodeId id) const
{
	std::vector<NodeId>::iterator iter = find(_vec_id.begin(), _vec_id.end(), id);
	return (iter != _vec_id.end()) ? iter - _vec_id.begin() + 1 : -1;
}

void RootCanvas::Render(report::ReportType type, NodeId id, TGraph* p_graph){

	if (! _b_rendering_started){
		initializeCanvas();
		_b_rendering_started = true;
	}

	int index = PadId(id);

	if (type == report::STATE && index >= 0){
		_p_pad_state->cd(index);
		_p_hist_state->Draw();
		p_graph->Draw("L");
	}

	if (type == report::RATE && index >= 0){
		_p_pad_rate->cd(index);
		_p_hist_rate->Draw();
		p_graph->Draw("L");
	}

	AddToCycle(id);
	if (IsCycleComplete() )
		_p_canvas->Update();
}

void RootCanvas::AddToCycle(NodeId id)
{
	// could be more efficient but rendering takes a lot of time anyway; this will not be the performance bottleneck
	std::vector<NodeId>::iterator iter = std::find(_vec_id.begin(),_vec_id.end(),id);
	if (iter == _vec_id.end())
		return;

	int i = iter - _vec_id.begin();
	_vec_scratch[i] = 1;
}

bool RootCanvas::IsCycleComplete(){
	std::vector<int>::iterator iter;
	iter = std::find(_vec_scratch.begin(),_vec_scratch.end(),0);
	if (iter == _vec_scratch.end()){
		for (iter = _vec_scratch.begin(); iter != _vec_scratch.end(); iter++)
			*iter = 0;
		return true;
	}
	else
		return false;
}

void RootCanvas::addNode(NodeId id)
{
	if (find(_vec_id.begin(),_vec_id.end(),id) == _vec_id.end())
			_vec_id.push_back(id);
//  else
//     nothing: multiple attempts to display the same node will simply be ignored

}

void RootCanvas::initializeCanvas(){

	_p_canvas = std::unique_ptr<TCanvas>(new TCanvas("MIIND_CANVAS","MIIND",CANVAS_X_DIMENSION,CANVAS_Y_DIMENSION));

	gStyle->SetOptStat(0);

    _p_pad_rate  = new TPad("rate", "", 0.05,0.05, 0.45, 0.95);
    _p_pad_state = new TPad("state","", 0.55,0.05, 0.95, 0.95);

    _p_hist_state = new TH2F("state_histo","", NUMBER_HISTO_CHANNELS ,_par_canvas._state_min,_par_canvas._state_max, NUMBER_HISTO_CHANNELS,_par_canvas._dense_min,_par_canvas._dense_max);
    _p_hist_rate  = new TH2F("rate_histo", "", NUMBER_HISTO_CHANNELS, _par_canvas._t_min,_par_canvas._t_max,NUMBER_HISTO_CHANNELS,_par_canvas._f_min,_par_canvas._f_max);

    this->SetMaximumDensity();
    this->SetMaximumRate();

    Number number_of_drawable_populations = _vec_id.size();
    _vec_scratch = std::vector<int>(number_of_drawable_populations,0.);
    _p_pad_rate->Draw();
    _p_pad_state->Draw();
    _p_pad_rate->cd();
    _p_pad_rate->Divide(1,number_of_drawable_populations, 1e-5, 1e-5);


    _p_pad_state->cd();
    _p_pad_state->Divide(1,number_of_drawable_populations, 1e-5, 1e-5);

    _p_canvas->cd();
    _p_canvas->Update();
}


void RootCanvas::SetMaximumDensity() const
{
 	if ( _par_canvas._dense_max > 0 ){
        int n_x      = _p_hist_state->GetXaxis()->GetNbins();
		int n_y      = _p_hist_state->GetYaxis()->GetNbins();
        double y_min = _p_hist_state->GetYaxis()->GetXmin();

        _p_hist_state->SetBins
		(
			n_x,
			_par_canvas._state_min,
			_par_canvas._state_max,
			n_y,
            y_min,
            _par_canvas._dense_max
	     );

 	}
}

void RootCanvas::SetMaximumRate() const
{
	if ( _par_canvas._f_max > 0 )
	{
		int n_x      = _p_hist_rate->GetXaxis()->GetNbins();
        int n_y      = _p_hist_rate->GetYaxis()->GetNbins();
        double y_min = _p_hist_rate->GetYaxis()->GetXmin();

        _p_hist_rate->SetBins
        (
        	n_x,
			_par_canvas._t_min,
			_par_canvas._t_max,
			n_y,
			y_min,
			_par_canvas._f_max
        );
	}
}

#endif // ENABLE_MPI


