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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifdef WIN32
#pragma warning(disable: 4996)
#endif

#include <TFile.h>
#include <TGraph.h>
#include <TKey.h>
#include <TROOT.h>
#include <TPostScript.h>
#include <TH2F.h>
#include <TAxis.h>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include "PopulistException.h"
#include "WorkFlowAnalyzer.h"

using namespace PopulistLib;

WorkFlowAnalyzer::WorkFlowAnalyzer
(
	const string& directory,
	const string& root_name
):
_directory(directory),
_root_name(root_name),
_p_file(InitFile()),
_vec_keys(InitKeys())
{
}

namespace {
	class  NotDesiredKey {
	public:

		NotDesiredKey(const PlotParameter& par_plot):_t_begin(par_plot._t_begin),_t_end(par_plot._t_end),_id(par_plot._id){}

		bool operator()(const GraphKey& key){
			return !(
						(
							key._id == this->_id && 
							key._time >= this->_t_begin &&
							key._time <= this->_t_end
						) || 
						((
							key._type == DynamicLib::RATEGRAPH &&  key._id == this->_id
						) && !(key._id == NodeId(0))) // never write out anything about NodeId(0)
					) ;
		}
	private:

		Time	_t_begin;
		Time	_t_end;
		NodeId	_id;
	};
}
namespace {

	class PlotKey {
	public:

		PlotKey(const string& directory, const string& plot_name, boost::shared_ptr<TFile> p_file):_directory(directory),_plot_name(plot_name),_p_file(p_file){}

		void operator()(const GraphKey& k){
			gROOT->SetBatch();
			gROOT->SetStyle("Plain");
			ostringstream strm;
			strm << _directory << "/" << _plot_name << "-" << k.Name()<< ".ps";
			TPostScript myps(strm.str().c_str(),111);
			TGraph* p_graph = (TGraph*)_p_file->Get(k.Name().c_str());
			TAxis* p_x_axis = p_graph->GetXaxis();
			double x_min = p_x_axis->GetXmin();
			double x_max = p_x_axis->GetXmax();
			TH2F hist(k.Name().c_str(),"",100,x_min,x_max,100,0.,5.);
			if (!p_graph)
				throw PopulistException(string("Could not retrieve file: ")+k.Name());

			p_graph->Draw("LP");
			myps.Close();
		}

	private:

		string _directory;
		string _plot_name;
		boost::shared_ptr<TFile> _p_file;
	};
}

bool WorkFlowAnalyzer::Plot
(
	const string&	plot_name
)
{	
	PlotParameter par_plot;

	bool b_parse = this->ParseFile(&par_plot);
	if (! b_parse)
		throw PopulistException("Parse file error");

	//select the keys and get the relevant graphs.
	vector<GraphKey>::iterator iter;
	vector<GraphKey> vec_result(_vec_keys.size());
	NotDesiredKey nokey(par_plot);

	vector<GraphKey>::iterator iend = remove_copy_if(_vec_keys.begin(), _vec_keys.end(), vec_result.begin(),nokey);
	vec_result.erase(iend + 1,vec_result.end());

	PlotKey plot(_directory, plot_name,_p_file);
	BOOST_FOREACH(const GraphKey& k, vec_result){
		plot(k);
	}

	return true;
}

bool WorkFlowAnalyzer::ParseFile(PlotParameter* p_plot)
{
	ifstream ifst(this->PlotParameterName().c_str() );
	if (ifst)
		ifst >> *p_plot;
	else {
		ofstream ofst(this->PlotParameterName().c_str());
		ofst << *p_plot;
	}

	return true;
}

vector<GraphKey> WorkFlowAnalyzer::InitKeys() const
{
	vector<GraphKey> _vec_key;
	TIter next(_p_file->GetListOfKeys());
	TKey* key;

	while((key =(TKey*)next())){
		GraphKey gkey(key->GetName());
		if (gkey._id != NodeId(0) )
			_vec_key.push_back(gkey);
	}

	return _vec_key;
}

boost::shared_ptr<TFile> WorkFlowAnalyzer::InitFile() const
{
	boost::shared_ptr<TFile> p_ret(	new TFile((_directory + string("/") + _root_name).c_str()));
	return p_ret;
}

string WorkFlowAnalyzer::PlotParameterName() const 
{
	typedef boost::tokenizer<boost::char_separator<char> > 
		tokenizer;
	boost::char_separator<char> sep(".");
	tokenizer tokens(_root_name, sep);
	tokenizer::iterator tok_iter = tokens.begin();

	string str_ret = _directory + string("/") + *tok_iter + string(".plotpar");
	return str_ret;
}
