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
#include <TClass.h>
#include <TFile.h>
#include <TGraph.h>
#include <TVectorT.h>
#include <TTree.h>
#include "DynamicLibException.h"
#include "RootHighThroughputHandler.h"

using namespace DynamicLib;

Time DynamicLib::RootHighThroughputHandler::_t_start = 0;
TTree* DynamicLib::RootHighThroughputHandler::_p_tree = 0;
bool DynamicLib::RootHighThroughputHandler::_is_recording = false;
bool DynamicLib::RootHighThroughputHandler::_is_first_time_slice_processed = false;
bool DynamicLib::RootHighThroughputHandler::_reinstate_node_graphs = false;
vector<double> DynamicLib::RootHighThroughputHandler::_vec_data = vector<double>(0);
TFile* DynamicLib::RootHighThroughputHandler::_p_file = 0;
TVectorD* DynamicLib::RootHighThroughputHandler::_p_array = 0;


RootHighThroughputHandler::RootHighThroughputHandler
(
	const string& file_name,
	bool have_state,
	bool reinstate_node_graphs
):
AbstractReportHandler
(
	file_name
)
{
	this->_reinstate_node_graphs = reinstate_node_graphs;
}

RootHighThroughputHandler::RootHighThroughputHandler
(
	const RootHighThroughputHandler& rhs
):
AbstractReportHandler(rhs)
{
}

void RootHighThroughputHandler::DetachHandler(const NodeInfo&)
{
	// write after the last array
	if (_p_array){
		_p_tree->Write();

		// if so desired, create node rate graphs
		// note that the file is closed by the first node so that the test for its existence is necessary
		if (_reinstate_node_graphs && _p_file)
			ReinstateNodeGraphs(_p_file->GetName());

	}

	// restore all static definitions
	delete _p_array;
	_p_array						= 0;
	_t_start						= 0;
	_is_recording					= false;
	_is_first_time_slice_processed	= false;
	_reinstate_node_graphs			= false;
	_vec_data						= vector<double>(0);
	_p_tree							= 0;

	if ( _p_file ) {
		// clean up
		_p_file->Close();
		delete _p_array;
		_p_array = 0;
		_p_file = 0;
	}
}

RootHighThroughputHandler* RootHighThroughputHandler::Clone() const
{
	return new RootHighThroughputHandler(*this);
}
bool RootHighThroughputHandler::Update()
{
	return true;
}

void RootHighThroughputHandler::InitializeHandler(const NodeInfo&)
{
	if (!_p_file)
		_p_file = new TFile(this->MediumName().c_str(),"RECREATE");
	if (_p_file->IsZombie() )
		throw DynamicLibException("Couldn't open root file");

}


bool RootHighThroughputHandler::WriteReport(const Report& report)
{

	if (report._id == NodeId(0) && _is_recording && !_is_first_time_slice_processed)
	{
		 // This is the first time that a complete time slice has been recorded, presumably the simulation time start
		_is_first_time_slice_processed = true;
		// so we should have a full fledged version of the data vector. We create a TArrayD which from now on will hold on data. From the vector we know the right size
		_p_array= new TVectorD(_vec_data.size() + 1);
		(*_p_array)[0] = _t_start;

		for (Index i = 0; i < _vec_data.size(); i++)
			(*_p_array)[i+1] = _vec_data[i];
	}

	if (report._id == NodeId(1) && !_is_recording)
	{
		_is_recording = true;
		_p_tree = new TTree("Activations","Times slices");
		_p_tree->Branch("slices","TVectorT<double>",&_p_array,32000,0);
	}



	if (! _is_first_time_slice_processed && _is_recording)
	{

		// Here we are adding events to the first time slice. We don't know how many there are, because here we don't know the size of the network
		// So we keep adding them to the event vector
		if (report._id != NodeId(0) )
			_vec_data.push_back(report._rate);

		// we also need to record the start time, since the slice can only be written once this step is complete, i.e. at the next report time.
		// But the time use when writing that slice must be the simulation start time
		_t_start = report._time;
	}

	if ( _is_first_time_slice_processed && _is_recording )
	{
		if (report._id == NodeId(1) ){
			// here we can write the TArrayD of the last time slice, as well as the time and start filling it up again
			_p_tree->Fill();
			// normal operation, just fill the TVectorD
			(*_p_array)[0] = report._time;
			(*_p_array)[report._id._id_value]=report._rate;
		}
		else
		{
			if (report._id != NodeId(0) ){
				// normal operation, just fill the TVectorD
				(*_p_array)[report._id._id_value]=report._rate;
			}
			else
				return true;
		}
	}
	return true;
}
RootHighThroughputHandler::~RootHighThroughputHandler()
{
}

bool RootHighThroughputHandler::ReinstateNodeGraphs(const char* p)
{
	const string file_name(p);
	//close to flush buffers and reopen to append
	_p_file->Close();
	delete _p_file;
	//closing the file has invalidated (destroyed) the tree
	_p_tree = 0;
	delete _p_array;
	_p_array = 0;

	_p_file = new TFile(file_name.c_str(),"UPDATE");

	TGraph* p_graph = (TGraph*)_p_file->Get("rate_1");
	if (p_graph){
		cout << "They are already in" << endl;
		return false;
	}

	Number number_of_nodes;
	Number number_of_slices;
	vector<double> vec_times;

	CollectGraphInformation(&vec_times,&number_of_nodes,&number_of_slices);

	StoreRateGraphs(vec_times,number_of_nodes,number_of_slices);

	_p_file->Close();	// the file object does not exist anymore, don't delete the pointer
	_p_file = 0;		// just set it to 0
	delete _p_array;
	_p_array = 0;

	return true;
}

void RootHighThroughputHandler::StoreRateGraphs(const vector<double>& vec_time, Number n_nodes, Number n_slices)
{
	TBranch* p_branch = _p_tree->GetBranch("slices");
	p_branch->SetAddress(&_p_array); //address of pointer!

	for (Index node = 0; node < n_nodes; node++ )
	{	
		Index id_node = node+1;
	
		vector<double> vec_rate;
		for (Index slice = 0; slice < n_slices; slice++ )
		{
			p_branch->GetEvent(slice);
			vec_rate.push_back((*_p_array)[id_node]);
		}
		if (vec_rate.size() != vec_time.size() )
			throw DynamicLibException("Inconsistency between times and rate");

		// Here we have all we need to create a Graph
		TGraph* p_graph = new TGraph(vec_rate.size(), &(vec_time[0]), &(vec_rate[0]));
		ostringstream stgr;
		stgr << "rate_" << id_node;
		p_graph->SetName(stgr.str().c_str());
		p_graph->Write();
		delete p_graph;
	}
}

void RootHighThroughputHandler::CollectGraphInformation(vector<double>* p_vec_time, Number* p_num_nodes, Number* p_num_slices){
	if (_p_tree)
		throw DynamicLibException("There is a TTree that shouldn't be there");

	_p_tree= (TTree*)_p_file->Get("Activations");
	if (!_p_tree)
		throw DynamicLibException("No valid TTree");

	*p_num_slices = static_cast<Number>(_p_tree->GetEntries());

	TBranch* p_branch = _p_tree->GetBranch("slices");
	p_branch->SetAddress(&_p_array); //address of pointer!

	for (Index i = 0; i < *p_num_slices; i++){
		p_branch->GetEvent(i);
		p_vec_time->push_back((*_p_array)[0]);
	}
	//exclude NodeId(0)
	*p_num_nodes = _p_array->GetNoElements() -1;
}
