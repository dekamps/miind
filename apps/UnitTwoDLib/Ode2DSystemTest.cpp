

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
#include <fstream>
#include <TwoDLib.hpp>

using namespace std;
using namespace TwoDLib;

BOOST_AUTO_TEST_CASE(SystemConstructionTest){
	Mesh mesh("aexpoverview.mesh");
	vector<double> v{-70.61e-3, -70.61e-3, -70.59e-3, -70.59e-3};
	vector<double> w{      0.0,      0.01,     0.01,   0.0};
	Quadrilateral reversal_bin(v,w);
	mesh.InsertStationary(reversal_bin);

	vector<Redistribution> reversal_map;
	for( unsigned int i = 0; i < mesh.NrStrips();i++){
		Coordinates from(i,0);
		Coordinates to(0,0);
		Redistribution m(from,to,1.0);
		reversal_map.push_back(m);
	}

	vector<Redistribution> reset_map;
	Ode2DSystem sys(mesh,reversal_map, reset_map);
}

BOOST_AUTO_TEST_CASE(MapTest){
	Mesh mesh("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mesh");
	vector<Redistribution> vec_dummy;
	vector<Redistribution> reset_map;
	Ode2DSystem sys(mesh,vec_dummy, vec_dummy);
	BOOST_REQUIRE(sys.Map(1,0) == 0);

	// Note that sys.Map(0,0) is undefined as no stationary point has been added to the Mesh
	// Adding the stationary point after the system has been defined does not change that.

	//However,
	vector<double> v{-64.99e-3, -64.99e-3, -65.01e-3, -65.01e-3};
	vector<double> w{      0.0,      0.01,     0.01,   0.0};
	Quadrilateral reversal_bin(v,w);
	mesh.InsertStationary(reversal_bin);

	Ode2DSystem sys2(mesh,vec_dummy,vec_dummy);
	// it exists in sys2
	BOOST_REQUIRE(sys2.Map(0,0) == 0);
	BOOST_REQUIRE(sys2.Map(1,0) == 1);

	// and if  we add another stationary
	vector<double> w2{ 0.01, 0.02, 0.02, 0.01};
	Quadrilateral reversal_bin2(v,w2);

	mesh.InsertStationary(reversal_bin2);
	Ode2DSystem sys3(mesh,vec_dummy,vec_dummy);
}

BOOST_AUTO_TEST_CASE(EvolutionTest){
	Mesh mesh("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mesh");

	vector<double> v{-64.99e-3, -64.99e-3, -65.01e-3, -65.01e-3};
	vector<double> w{      0.0,      0.01,     0.01,   0.0};
	Quadrilateral reversal_bin(v,w);
	mesh.InsertStationary(reversal_bin);

	vector<Redistribution> reversal_map;
	// 1 is crucial. If 0 is in there, the reversal bin will be wiped out
	for( unsigned int i = 1; i < mesh.NrStrips();i++){
		Coordinates from(i,0);
		Coordinates to(0,0);
		Redistribution m(from,to,1.0);
		reversal_map.push_back(m);
	}

	vector<Redistribution> reset_map;
	Ode2DSystem sys(mesh,reversal_map, reset_map);
	sys.Initialize(1,0);

	std::ofstream ofst10("condres_10");
	for (int i = 0; i < 10; i++)
		sys.Evolve();
	sys.Dump(ofst10);

	std::ofstream ofst1000("condres_1000");
	for (int i = 0; i < 1000; i++)
		sys.Evolve();
	sys.Dump(ofst1000);
}

BOOST_AUTO_TEST_CASE(DumpTest){
	Mesh mesh("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mesh");

	vector<double> v{-64.99e-3, -64.99e-3, -65.01e-3, -65.01e-3};
	vector<double> w{      0.0,      0.01,     0.01,   0.0};
	Quadrilateral reversal_bin(v,w);
	mesh.InsertStationary(reversal_bin);

	vector<Redistribution> reversal_map;
	// 1 is crucial. If 0 is in there, the reversal bin will be wiped out
	for( unsigned int i = 1; i < mesh.NrStrips();i++){
		Coordinates from(i,0);
		Coordinates to(0,0);
		Redistribution m(from,to,1.0);
		reversal_map.push_back(m);
	}

	vector<Redistribution> reset_map;
	Ode2DSystem sys(mesh,reversal_map, reset_map);
	sys.Initialize(100,0);

	// inspect these files with the visualize function in the cond.py script
	std::ofstream ofst11("condres11");
	for (int i = 0; i < 11; i++)
		sys.Evolve();
	sys.Dump(ofst11);
	sys.Evolve();

	std::ofstream ofst12("condres12");
	sys.Dump(ofst12);
}

BOOST_AUTO_TEST_CASE(ResetTest){
	Mesh mesh("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mesh");

	vector<double> v{-64.99e-3, -64.99e-3, -65.01e-3, -65.01e-3};
	vector<double> w{      0.0,      0.01,     0.01,   0.0};
	Quadrilateral reversal_bin(v,w);
	mesh.InsertStationary(reversal_bin);

	vector<Redistribution> reversal_map;
	// 1 is crucial. If 0 is in there, the reversal bin will be wiped out
	for( unsigned int i = 1; i < mesh.NrStrips();i++){
		Coordinates from(i,0);
		Coordinates to(0,0);
		Redistribution m(from,to,1.0);
		reversal_map.push_back(m);
	}

	std::ifstream ifstres("condee2a5ff4-0087-4d69-bae3-c0a223d03693.res");
	vector<Redistribution> reset_map = ReMapping(ifstres);
	Ode2DSystem sys(mesh,reversal_map, reset_map);
	sys.Initialize(100,0);

	std::ofstream ofstreset("reset37");
	for (int i = 0; i < 37; i++)
		sys.Evolve();
	sys.Dump(ofstreset);
}
