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
#include <TFile.h>
#include <TGraph.h>
#include <TFileIter.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TF1.h>
#include <boost/bind.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include "RootFileInterpreter.h"
#include "DynamicLibException.h"

using namespace DynamicLib;

RootFileInterpreter::RootFileInterpreter(TFile& file):
_p_file(&file),
_n_rate(0),
_n_state(0)
{		
	boost::char_separator<char> sep("_");

	TFileIter readObj(&file);
	int n_objects = readObj.TotalKeys();

	TObject* nextObject;
	for( readObj = 0; int(readObj) < n_objects; ++readObj){
		nextObject = *readObj;
		string s = nextObject->GetName();
		boost::tokenizer<boost::char_separator<char> > tokens(s, sep);
		boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();
		OrderKeys(it,s);
	}
}

vector<Time> RootFileInterpreter::StateTimes(NodeId id) const
{
	vector<Time> vec_times(_vec_times[id._id_value].size());
	vector<Time>::iterator titer = vec_times.begin();
	for( vector<timename>::const_iterator iter = _vec_times[id._id_value].begin(); iter !=  _vec_times[id._id_value].end(); iter++,titer++)
		*titer = iter->first;

	return vec_times;
}

void RootFileInterpreter::OrderKeys(boost::tokenizer<boost::char_separator<char> >::iterator it, const string& name)
{
	if (*it == "grid" ){
		++_n_state;
		Index pop_id = atoi((++it)->c_str());
		Time t = atof((++it)->c_str());

		while (_vec_times.size() <= pop_id)
			_vec_times.push_back(vector<timename>(0));
		_vec_times[pop_id].push_back(timename(t,name));


	} else {
		if (*it == "rate")
			++_n_rate;
	}
}

TGraph* RootFileInterpreter::GetStateGraph(NodeId id, Time t ){
	assert( id._id_value < static_cast<int>(_vec_times.size()) + 1);
	vector<timename> vec_times = _vec_times[id._id_value];
	double diff = fabs(vec_times[0].first - t);

	Index ind = 0;
	for( vector<timename>::iterator iter = vec_times.begin(); iter != vec_times.end(); iter++)
		if (fabs(iter->first - t) < diff){
			ind = iter - vec_times.begin();
			diff = fabs(iter->first-t);
		}
	TGraph* p_graph = (TGraph*)_p_file->Get(vec_times[ind].second.c_str());
	return p_graph;
}

TGraph* RootFileInterpreter::GetRateGraph(NodeId id)
{
	ostringstream ost;
	ost << "rate_" << id._id_value;
	TGraph* p_ret = (TGraph*)_p_file->Get(ost.str().c_str());
	return p_ret;
}

Rate RootFileInterpreter::ExtractFiringRate(NodeId id, DynamicLib::Time t_begin_fit, DynamicLib::Time t_end_fit) 
{
	TGraph* p_rate_graph = GetRateGraph(id);

	Axis_t t_begin = t_begin_fit;
	Axis_t t_end   = t_end_fit;

	p_rate_graph->Fit("pol1","SQ","",t_begin, t_end);
	TF1* pf = p_rate_graph->GetFunction("pol1");

	Double_t v0 = pf->GetParameter(0);

	return v0;
}
