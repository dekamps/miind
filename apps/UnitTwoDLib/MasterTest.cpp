// Copyright (c) 2005 - 2015 Marc de Kamps
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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <TwoDLib.hpp>
#include <fstream>


using namespace TwoDLib;

BOOST_AUTO_TEST_CASE(MasterTest){
/*	Mesh mesh("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mesh");

	vector<double> v{-64.99e-3, -64.99e-3, -65.01e-3, -65.01e-3};
	vector<double> w{      0.0,      0.01,     0.01,   0.0};
	Quadrilateral reversal_bin(v,w);
	mesh.InsertStationary(reversal_bin);

	vector<Redistribution> reversal_map;
	// 1 is crucial. If 0 is in there, the reversal bin will be wiped out
	for( unsigned int i = 1; i < mesh.NrQuadrilateralStrips();i++){
		Coordinates from(i,0);
		Coordinates to(0,0);
		Redistribution m(from,to,1.0);
		reversal_map.push_back(m);
	}

	std::ifstream ifstres("condee2a5ff4-0087-4d69-bae3-c0a223d03693.res");

	vector<Redistribution> reset_map = ReMapping(ifstres);
	Ode2DSystem sys(mesh,reversal_map, reset_map);
	sys.Initialize(0,0);
	TransitionMatrix mat("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mat");

	vector<TransitionMatrix> vec_mat;
	vec_mat.push_back(mat);

	MasterParameter N = {100};
	Master master(sys,vec_mat,N);

	double rate = 1000.0;
	vector<double> vec_rates;
	vec_rates.push_back(rate);

	double t_step = mesh.TimeStep();
	int nr_steps = 400;
	std::ostringstream ost;
	ost << nr_steps;
	std::ofstream ofstmaster("master_" + ost.str() );
	for (int i = 0; i < nr_steps; i++){
		sys.Evolve();
		master.Apply(t_step,vec_rates);
		sys.RedistributeProbability();
	}
	sys.Dump(ofstmaster);*/
}
