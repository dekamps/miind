// Copyright (c) 2005 - 2009 Marc de Kamps
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
#include "NetworkedRootReportHandler.h"
#include "DynamicLib/RootReportHandlerPrivate.h"
#include "DynamicLib/DynamicLibException.h"
//#include "DynamicLib/LocalDefinitions.h"
#include <cstdlib>




#include <TApplication.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TNtuple.h>
#include <TStyle.h>
#include <TSystem.h>

using namespace ROOT;


using namespace std;
using namespace DynamicLib;




namespace {
// The following objects are also defined in the CMRootReportHandler

	// need one global application object
//	TApplication APPLICATION("application",0,0);
	TH2F	HISTO_RATE
		(
			"rate_histo",
			"",
			NUMBER_HISTO_CHANNELS,
			T_MIN,
			T_MAX,
			NUMBER_HISTO_CHANNELS,
			F_MIN,
			F_MAX
		);

	TH2F	HISTO_STATE 
		(
			"state_histo",
			"",
			NUMBER_HISTO_CHANNELS,
			0,
			500,
			NUMBER_HISTO_CHANNELS,
			-0.5,
			0.5
		);
}


TFile*   NetworkedRootReportHandler::_p_file      = 0;
TCanvas* NetworkedRootReportHandler::_p_canvas    = 0;
TNtuple* NetworkedRootReportHandler::_p_tuple     = 0;
TPad*	 NetworkedRootReportHandler::_p_pad_rate  = 0;
TPad*    NetworkedRootReportHandler::_p_pad_state = 0;
TPad*    NetworkedRootReportHandler::_p_fc_state = 0;
vector<NodeId> NetworkedRootReportHandler::_list_nodes(0);
vector<NodeId> NetworkedRootReportHandler::_vector_id(0);
namespace {  // added anonymous namespace to prevent conflict with similar CMRootReportHanlder definition
TGraph* eq_graph    = 0;
}

NetworkedRootReportHandler::NetworkedRootReportHandler
	(
		const string& file_name,
		bool b_canvas,
		bool b_file,
		double t_max,
		double rate_max,
		double v_max,
		double density_max
	):
AbstractReportHandler(file_name),
_p_current_rate_graph	(0),
_p_current_state_graph	(0),
_b_canvas				(b_canvas),
_b_file					(b_file),
_nr_reports				(0),
_index_pad				(-1),
_rate_max               (rate_max),
_t_max                  (t_max),
_v_max                  (v_max),
_density_max            (density_max)
{

}

NetworkedRootReportHandler::NetworkedRootReportHandler(const NetworkedRootReportHandler& rhs):
AbstractReportHandler(rhs.MediumName()),
_p_current_rate_graph	(0),
_p_current_state_graph	(0),
_b_canvas				(rhs._b_canvas),
_b_file					(rhs._b_file),
_nr_reports				(0),
_index_pad				(-1),
_rate_max               (rhs._rate_max),
_t_max                  (rhs._t_max),
_v_max                  (rhs._v_max),
_density_max            (rhs._density_max)
{
	if (rhs._p_current_rate_graph)
		throw DynamicLibException(STR_HANDLER_STALE);
}

NetworkedRootReportHandler::~NetworkedRootReportHandler()
// 11-07-2007: test on p_tuple by Volker Baier
{

return;
	if ( _p_file ){
		if ( _p_tuple )
			_p_tuple->Write();
		_p_file->Close();
		delete _p_file;
		_p_file = 0;
	}

	if ( _p_canvas){
		delete _p_canvas;
		_p_canvas = 0;
	}
}

bool NetworkedRootReportHandler::WriteReport(const Report& report)
{

	if((int)report._time%2==1)return true;
	int time = report._time/2;
	if (_nr_reports == 0)
	{
		if (_t_max != 0)
			this->SetTimeRange(0,_t_max);

		_p_current_rate_graph = new TGraph;
		_p_current_state_graph = new TGraph;
		_p_psi_graph = new TGraph;
		_p_f_graph = new TGraph;
		_p_c_graph = new TGraph;
		_p_g_graph = new TGraph;

		_p_current_rate_graph->SetLineWidth(2);
		_p_current_rate_graph->SetMarkerStyle(7);
		
		_p_current_state_graph->SetLineColor(8);
		_p_current_state_graph->SetLineWidth(2);

		_p_g_graph->SetLineColor(9);
		_p_g_graph->SetLineWidth(2);

		_p_psi_graph->SetMarkerStyle(5);
		_p_psi_graph->SetLineWidth(2);
		_p_psi_graph->SetLineStyle(2);

		_p_f_graph->SetLineColor(46);
		_p_f_graph->SetLineWidth(2);

		_p_c_graph->SetLineColor(38);
		_p_c_graph->SetLineWidth(2);
		//_p_c_graph->SetLineStyle(2);

		ostringstream stream ;
		stream << "rate_" << report._id;
		_p_current_rate_graph->SetName(stream.str().c_str());

		vector<NodeId>::iterator 
			iter = find
				(
					_vector_id.begin(),
					_vector_id.end(),
					report._id
				);

		if ( iter !=  _vector_id.end() )
			_index_pad = static_cast<int>(iter - _vector_id.begin()) + 1;
		// else
			// ignore

		/*_p_psi_graph->SetPoint
		(
			0,
			0,
			0
		);

		_p_f_graph->SetPoint
		(
			0,
			0,
			0
		);

		_p_c_graph->SetPoint
		(
			0,
			0,
			0
		);*/
	}

	string message = report._log_message;
	//cout << message << endl;

	if(message == "") return true;

	char c = message[0];
	message = message.substr(1);
	if(c == 'm'){


		_p_current_rate_graph->SetPoint
		(
			_nr_reports,
			time,
			report._rate
		);

		string buf; // Have a buffer string
		stringstream ss(message); // Insert the string into a stream

		vector<string> tokens; // Create vector to hold our words

		while (ss >> buf)
			tokens.push_back(buf);


		//cout << report._log_message << " " << tokens[0] << " " << tokens[1] << endl;


		double equilibrium = atof(tokens[0].c_str());
		_p_current_state_graph->SetPoint
		(
			_nr_reports,
			time,
			0.5
		);

		_p_g_graph->SetPoint
		(
			_nr_reports,
			time,
			0
		);


		double psi = atof(tokens[1].c_str());
		_p_psi_graph->SetPoint
		(
			_nr_reports,
			psi,
			report._rate-equilibrium
		);

		double fd = atof(tokens[2].c_str());
		//cout << "f: " <<fd << endl;
		_p_f_graph->SetPoint
		(
			_nr_reports,
			time,
			fd
		);

		double cd = atof(tokens[3].c_str());
		//cout << "c: " <<cd << endl;
		_p_c_graph->SetPoint
		(
			_nr_reports,
			time,
			cd
		);
	}
	else
	{
	}
	_nr_reports++;
	return true;
}



NetworkedRootReportHandler* NetworkedRootReportHandler::Clone() const
{
	// Cloning happens at configure time.
	// Now is the time to divide the pads

	if (_p_canvas && ! _p_pad_rate )
		InitializeCanvas();

	return new NetworkedRootReportHandler(*this);
}


TGraph* NetworkedRootReportHandler::ConvertAlgorithmGridToGraph
(	
	const Report& report
) const
{

	vector<double> vector_of_grid_values  = report._grid.ToStateVector();

	// if the Report does not contain a filled AlgorithmGrid, no Graph can be made
	if ( vector_of_grid_values.size() == 0 )
		return 0;


	vector<double> vector_of_state_interpretation = report._grid.ToInterpretationVector();

	TGraph* p_state_graph = new TGraph;
	string title = string("grid_") + report.Title();
	p_state_graph->SetName(title.c_str());

	assert( vector_of_grid_values.size() == vector_of_state_interpretation.size() );
	for
	( 
		vector<double>::iterator iter = vector_of_grid_values.begin();
		iter != vector_of_grid_values.end();
		iter++
	)
	{
		int n_index = static_cast<int>(iter - vector_of_grid_values.begin());

		p_state_graph->SetPoint
		(
			n_index,
			vector_of_state_interpretation[n_index],
			vector_of_grid_values[n_index]
		);

	}

	return p_state_graph;
}

bool NetworkedRootReportHandler::BelongsToAnAlgorithm() const
{
	return (_p_current_rate_graph != 0);
}

bool NetworkedRootReportHandler::IsStateWriteMandatory() const
{
	return _b_file;
}

bool NetworkedRootReportHandler::HasANoneTrivialState(const Report& report) const
{
	return true;
}

void NetworkedRootReportHandler::AddNodeToCanvas(NodeId id) const
{
	_vector_id.push_back(id);
}

void NetworkedRootReportHandler::InitializeCanvas() const
{
	this->SetMaximumDensity();
	this->SetMaximumRate();
	gStyle->SetOptStat(0);
	gStyle->SetFillColor(0);

	_p_canvas->SetTitle("Fundamentalists and Chartist Marker Simulation");

	eq_graph = new TGraph();

	_p_pad_rate  = new TPad("rate", "", 0.0,0.5, 1, 1);
	
	_p_pad_state = new TPad("state","", 0.0,0.0, 0.5, 0.5);

	_p_fc_state = new TPad("fc","", 0.5,0.0, 1, 0.5);

	_p_pad_rate->SetTitle("Stock Price");
	
	_p_pad_rate->SetGrid();
	//_p_pad_state->SetGrid();
	_p_fc_state->SetGrid();

	int number_of_drawable_populations = _vector_id.size();

	_p_pad_rate->Draw();
	_p_pad_state->Draw();
	_p_fc_state->Draw();
	_p_pad_rate->cd();



	_p_pad_state->cd();

	_p_fc_state->cd();
	
	_p_canvas->cd();
	_p_canvas->Update();

	_p_canvas->SetFillColor(0);
	_p_canvas->SetGrid();

HISTO_RATE.SetTitle("Price");
	
	HISTO_RATE.SetXTitle("Time step");
	HISTO_RATE.SetYTitle("Price");
	HISTO_RATE.SetLabelSize(0.02, "X");
	HISTO_RATE.SetLabelSize(0.02, "Y");

	_p_pad_rate->SetFillColor(0);

}

void NetworkedRootReportHandler::ToRateCanvas(int n_index) 
{

}

bool NetworkedRootReportHandler::Update()
{
	//gSystem->Sleep ( 5 );

	if (_p_canvas && _index_pad > -1)
	{
		_p_pad_rate->cd();
		HISTO_RATE.Draw();
		_p_current_rate_graph->Draw("AL");
		_p_current_state_graph->Draw("L");
		_p_g_graph->Draw("L");


		_p_pad_state->cd();
		 //HISTO_STATE.Draw();
		_p_psi_graph->Draw("AL");


		_p_fc_state->cd();
		// HISTO_STATE.Draw();
		_p_c_graph->Draw("AL");
		_p_f_graph->Draw("L");
		

		_p_canvas->cd();
		_p_canvas->Update();
		

	}

	return true;
}

void NetworkedRootReportHandler::InitializeHandler(const NodeInfo& info)
{
	// Purpose: this function will be called by DynamicNode upon configuration. This means that if there is no canvas yet, and it is desired,
	// that now is the time to create it. This works under the assumption that no two DynamicNetwork simulations are running at the same time.
	// Idea is that several handlers can be created, but that no competition between the ROOT resources, which contain globals and statics, takes place
	// Assumptions: No two DynamicNetwork simulations are running at the same time
	// Author: Marc de Kamps
	// Date: 26-08-2005

	if ( ! _p_file ){
		_p_file = 
			new TFile
			(
				this->MediumName().c_str(),
				"RECREATE"
			);

		if ( _p_file->IsZombie() )
			throw DynamicLibException(STR_ROOT_FILE_OPENED_FAILED);

		_p_tuple =
			new TNtuple
			(
				"infotuple",
				"node info",
				"id:x:y:z:f"
			);

	}

	WriteInfoTuple(info);

	if ( _b_canvas ){

		if (! _p_canvas){
			_p_canvas = 
			new TCanvas
			(
				CANVAS_NAME.c_str(), 
				CANVAS_TITLE.c_str(),
				CANVAS_X_DIMENSION,
				CANVAS_Y_DIMENSION
			);
			InitializeCanvas();
		}
	}

}

void NetworkedRootReportHandler::WriteInfoTuple
(
	const NodeInfo& info
)
{
	_p_tuple->Fill
		(
			static_cast<Float_t>(info._id._id_value),
			info._position._x,
			info._position._y,
			info._position._z,
			info._position._f
		);
	_list_nodes.push_back(info._id);
}

void NetworkedRootReportHandler::DetachHandler
(
	const NodeInfo& info
 )
{
	// Purpose: this function will be called upon DynamicNode destruction. This works under the assumption that no isolated DynamicNodes
	// exist which are associated with an open NetworkedRootReportHandler. 
	// Author: Marc de Kamps
	// Date: 26-08-2005

	RemoveFromNodeList(info._id);

	if (_p_current_rate_graph)
	{
		_p_current_rate_graph->Write();
		delete _p_current_rate_graph;
		_p_current_rate_graph = 0;
	}

	if ( _list_nodes.empty() )
		GlobalCleanUp();
}

void NetworkedRootReportHandler::SetMaximumDensity() const
{
	if ( _density_max > 0 )
	{
		int n_x      = HISTO_STATE.GetXaxis()->GetNbins();
		double x_min = HISTO_STATE.GetXaxis()->GetXmin();
		int n_y      = HISTO_STATE.GetYaxis()->GetNbins();
		double y_min = HISTO_STATE.GetYaxis()->GetXmin();
		
		HISTO_STATE.SetBins
		(
			n_x,
			x_min,
			_v_max,
			n_y,
			y_min,
			_density_max
		);

	}
}

void NetworkedRootReportHandler::SetMaximumRate() const
{
	if ( _rate_max > 0 )
	{
		int n_x      = HISTO_RATE.GetXaxis()->GetNbins();
		double x_min = HISTO_RATE.GetXaxis()->GetXmin();
		int n_y      = HISTO_RATE.GetYaxis()->GetNbins();
		double y_min = HISTO_RATE.GetYaxis()->GetXmin();
		
		HISTO_RATE.SetBins
		(
			n_x,
			x_min,
			_t_max,
			n_y,
			y_min,
			_rate_max
		);
	}
}

void NetworkedRootReportHandler::RemoveFromNodeList(NetLib::NodeId id)
{
	vector<NodeId>::iterator iter = 
		find
		(
			_list_nodes.begin(),
			_list_nodes.end(),
			id
		);

	if (iter == _list_nodes.end() )
		throw DynamicLibException("Can't locate NodeId during detaching handler");

	_list_nodes.erase(iter);
}

void NetworkedRootReportHandler::GlobalCleanUp()
{
	_p_tuple->Write();
	_p_file->Close();
return;
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

void NetworkedRootReportHandler::SetDensityRange
(
	DynamicLib::Density d_min, 
	DynamicLib::Density d_max
)
{
		int n_x      = HISTO_STATE.GetXaxis()->GetNbins();
		double x_min = HISTO_STATE.GetXaxis()->GetXmin();
		int n_y      = HISTO_STATE.GetYaxis()->GetNbins();
		
		HISTO_STATE.SetBins
		(
			n_x,
			x_min,
			_v_max,
			n_y,
			d_min,
			d_max
		);
}

void NetworkedRootReportHandler::SetFrequencyRange
(
	Rate r_min,
	Rate r_max
)
{
		int n_x      = HISTO_RATE.GetXaxis()->GetNbins();
		double x_min = HISTO_RATE.GetXaxis()->GetXmin();
		int n_y      = HISTO_RATE.GetYaxis()->GetNbins();
		
		HISTO_RATE.SetBins
		(
			n_x,
			x_min,
			_v_max,
			n_y,
			r_min,
			r_max
		);
}

void NetworkedRootReportHandler::SetTimeRange(DynamicLib::Time t_min, DynamicLib::Time t_max)
{
		int n_x      = HISTO_RATE.GetXaxis()->GetNbins();
		int n_y      = HISTO_RATE.GetYaxis()->GetNbins();
		double y_min = HISTO_RATE.GetYaxis()->GetXmin();
		double y_max = HISTO_RATE.GetYaxis()->GetXmax();
		
		HISTO_RATE.SetBins
		(
			n_x,
			t_min,
			t_max,
			n_y,
			y_min,
			y_max
		);

}

void NetworkedRootReportHandler::SetPotentialRange
(
	Potential v_min,
	Potential v_max
)
{
		int n_x      = HISTO_STATE.GetXaxis()->GetNbins();
		double y_min = HISTO_STATE.GetYaxis()->GetXmin();
		double y_max = HISTO_STATE.GetYaxis()->GetXmax();
		int n_y      = HISTO_STATE.GetYaxis()->GetNbins();
		
		HISTO_STATE.SetBins
		(
			n_x,
			v_min,
			v_max,
			n_y,
			y_min,
			y_max
		);
}



