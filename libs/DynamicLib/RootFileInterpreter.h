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
#ifndef _CODE_LIBS_DYNAMICLIB_ROOTFILEINTERPRETER_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_ROOTFILEINTERPRETER_INCLUDE_GUARD


#include <vector>
#include "../NetLib/NetLib.h"
#include "BasicDefinitions.h"

class TFile;
class TGraph;

using NetLib::NodeId;
using std::vector;

namespace DynamicLib {

	//! Provides access to names of objects in the root file. In files produced by the RootReportHandler, there
	//! are rate graphs and state graphs. In files produced with the RootHighThroughputHandler, there are rate graphs
	//! and an activity tree. 

	class RootFileInterpreter {
	public:

		//! construct object
		RootFileInterpreter(TFile& );
		
		//! The number of rate graphs, should be equal to the number of nodes in the simulation.
		Number NumberOfRateGraphs();

		//! Get the rate graph indicated by the NodeId. Valid node numbers are 1 , ..., NumberofRateGraphs().
		//! Returns the null pointer if the graph can not be found.
		TGraph* GetRateGraph(NodeId);

		//! Give a list of times for which state graphs exist for the given NodeId. Times returned by this list are valid
		//! for use in GetStateGraph.
		vector<Time> StateTimes(NodeId) const;

		//! Will return the rate graph of the node indicated by the id, at the time given by the time. If the Root file
		//! is created by a RootHighThroughput object, this function will throw an exception. If the time provided
		//! does not match the time of a state graph exactly, the function will attempt
		//! to match the time of the state graph as closely as possible. The user must check the graph name to see if the
		//! match is close enough.
		TGraph* GetStateGraph(NodeId, Time);

		//! If this is true, do not attempt to get state graphs.
		bool IsHighThroughput() const;

		//! A constant function is fit to a rate graph over the interval specified by the begin and end time
		Rate ExtractFiringRate(NodeId, Time, Time);

	private:

		typedef pair<Time,string> timename;

		void OrderKeys(boost::tokenizer<boost::char_separator<char> >::iterator, const string&);

		TFile* _p_file;
		Number _n_rate;
		Number _n_state;

		vector< vector<timename> > _vec_times;
	};
}

#endif // include guard