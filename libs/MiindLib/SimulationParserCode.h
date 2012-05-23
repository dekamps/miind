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
#ifndef _CODE_LIBS_MIINDLIB_SIMULATIONPARSERCODE_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_SIMULATIONPARSERCODE_INCLUDE_GUARD

#include <fstream>
#include <TFile.h>
#include <TGraph.h>
#include <TPostScript.h>
#include "../DynamicLib/DynamicLib.h"
#include "../PopulistLib/PopulistLib.h"
#include "SimulationParser.h"
#include "SimulationBuilder.h"


using namespace std;

namespace MiindLib {

	SimulationParser::SimulationParser()
	{
	}

	bool SimulationParser::GenerateXMLFile(const string& file_name, const string& type)
	{
		if ( type == "double"){
			SimulationBuilder<double> builder;
			builder.GenerateExample(file_name);
			return true;
		}
		if ( type == "pop" || type == "ou" ){
			SimulationBuilder<PopulistLib::PopulationConnection> builder;
			builder.GenerateExample(file_name);
			return true;
		}

		throw PopulistException("Unknown weightype in SimulationParser");
	}

	bool SimulationParser::ExecuteSimulation
	(
		const string& file_name,
		bool b_batch
	)
	{
		ifstream ifst(file_name.c_str());
		if (! ifst ){
			cout << "Couldn't open: " << file_name << endl;
			return false;
		}

		string dummy;
		ifst >> dummy;

		if ( dummy != "<Simulation>"  && dummy != "﻿<Simulation>"){ // allow for possible BOM
			cout << "File doesn't contain a proper simulation" << endl;
			return false;
		}

		ifst >> dummy;

		if ( dummy == "<WeightType>double</WeightType>" ){
			SimulationBuilder<double> build;
			bool b_res =build.BuildSimulation(ifst, b_batch);
			_vec_canvas_ids = build.CanvasIds();
			_name_simulation_result = build.SimulationFileName();
			return b_res;
		}

		if ( dummy == "<WeightType>Pop</WeightType>" ){
			SimulationBuilder<PopulationConnection> build;
			bool b_res = build.BuildSimulation(ifst, b_batch);
			_vec_canvas_ids = build.CanvasIds();
			_name_simulation_result = build.SimulationFileName();
			return b_res;
		}

		return true;
	}

	bool SimulationParser::Analyze(const string& ){


		TFile file(_name_simulation_result.c_str());
		DynamicLib::RootFileInterpreter inter(file);

		vector<NodeId>::iterator it;
		for
		(
			it =  _vec_canvas_ids.begin(); 
			it != _vec_canvas_ids.end(); 
			it++
		){
			vector<Time> vec = inter.StateTimes(*it);
			Time t1 = vec[0];
			Time t2 = vec[1];
		
			TGraph* p1 = inter.GetStateGraph(NodeId(*it),t1);
			TPostScript ps1((string("test/") + string(p1->GetName()) + string(".ps")).c_str());
			p1->Draw("AL");
			ps1.Close();
			TGraph* p2 = inter.GetStateGraph(NodeId(*it),t2);
			TPostScript ps2((string("test/") + string(p2->GetName()) + string(".ps")).c_str());
			p2->Draw("AL");
			ps2.Close();
		}
		return true;
	}
}

#endif // include guard