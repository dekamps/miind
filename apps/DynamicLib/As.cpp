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
#include <iostream>
#include <fstream>

#include "../PopulistLib/PopulistLib.h"
#include <MiindLib/MiindLib.h>

using PopulistLib::Pop_Network;
using DynamicLib::EXCITATORY;
using DynamicLib::INHIBITORY;
using DynamicLib::SimulationRunParameter;
using DynamicLib::RootReportHandler;
using PopulistLib::InitialDensityParameter;

using std::cout;
using std::endl;
using std::ofstream;

Rate Start(Time t){
	return (t < 0.10) ? 10.0: 0.0;
}


int main(int argc, char* argv[])
{
    cout << "Large synapse example" << endl;

	Pop_Network net;
	RateFunctor<OU_Connection> func(Start);

	NodeId id_start = net.AddNode(func,EXCITATORY);

	const PopulistSpecificParameter
		specific
		(
			-0.005,
			500,
			1,
			InitialDensityParameter(0.,0.),
			1.1,
			"NumericalZeroLeakEquations"
		);

	PopulationParameter par(20e-3, 0.0,0.0,2e-3,20e-3);
	PopulistParameter par_pop(par,specific);

	PopulationAlgorithm alg(par_pop);

	NodeId id_e = net.AddNode(alg,EXCITATORY);
	NodeId id_i = net.AddNode(alg,INHIBITORY);


	RootReportHandler handler("largesynapse.root",true,true);
	handler.SetPotentialRange(-0.005,0.020);
	handler.SetFrequencyRange(0.,30.);
	handler.SetDensityRange(-0.01,3);
	handler.SetTimeRange(0.,0.3);



	handler.AddNodeToCanvas(id_e);
	handler.AddNodeToCanvas(id_i);
//	handler.AddNodeToCanvas(id_start);


	OU_Connection con_r_e(1.0,0.025);
	net.MakeFirstInputOfSecond(id_start,id_e,con_r_e);
	net.MakeFirstInputOfSecond(id_start,id_i,con_r_e);

	double prob = 0.05;
	OU_Connection con_r_ee(10000*prob,2e-3);
	OU_Connection con_r_ei(10000*prob,2e-3);
	OU_Connection con_r_ie(2500*prob,-9e-3);
	OU_Connection con_r_ii(2500*prob,-9e-3);

	net.MakeFirstInputOfSecond(id_e,id_e,con_r_ee);
	net.MakeFirstInputOfSecond(id_e,id_i,con_r_ei);
	net.MakeFirstInputOfSecond(id_i,id_e,con_r_ie);
	net.MakeFirstInputOfSecond(id_i,id_i,con_r_ii);
	
	SimulationRunParameter
		par_run
		(
			handler,
			10000000,
			0.0,
			0.2,
			1e-4,
			1e-4,
			1e-5,
			"log"
		);

	net.ConfigureSimulation(par_run);

	net.Evolve();

	return 0;
}
