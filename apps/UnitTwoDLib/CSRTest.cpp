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
#include <vector>
#include <fstream>
#include <iostream>
#include "FixtureOde2DSystemGroup.hpp"


using namespace std;
using namespace TwoDLib;

BOOST_FIXTURE_TEST_CASE(CSRMatrixTest, FixtureOde2DSystemGroup)
{
	// generate transition matrix for mesh 1
	std::ofstream ofst1("mat1.mat");
	ofst1 << "0.1 0.0\n"; // this jump size is immaterial and will not be used
	ofst1 << "10;0,0;1,0:1.0;\n";
	ofst1 << "10;0,1;1,1:1.0;\n";
	ofst1 << "10;1,0;2,0:0.4;2,1:0.6;\n";
	ofst1.close();
	
	std::ofstream ofst2("mat2.mat");
	ofst2 << "0.1 0.0\n"; // this jump size is immaterial and will not be used
	ofst2 << "10;0,0;1,0:0.3;1,1:0.3;1,2:0.4\n";
	ofst2 << "10;1,0;2,0:0.4;2,1:0.6;\n";
	ofst2.close();
	
	TransitionMatrix mat1("mat1.mat");
	TransitionMatrix mat2("mat2.mat");
	
	std::vector<std::vector<Redistribution> > vec_dummy;
	std::vector<Mesh> vec_mesh {_mesh1, _mesh2 };
       	Ode2DSystemGroup group(vec_mesh,vec_dummy,vec_dummy);
	std::cout << "yipta" << std::endl;
	//group.Initialize(0,0,0);
	//group.Initialize(1,0,0);
	/*
	CSRMatrix csr1(mat1,group,0);
	CSRMatrix csr2(mat2,group,1);

	std::vector<double> dydt(group.Mass().size(),0.0);
	csr1.MVMapped(dydt,group.Mass(),1000.);
	csr2.MVMapped(dydt,group.Mass(),800.);

	BOOST_CHECK( dydt[0] == -1000.);
	BOOST_CHECK( dydt[2] ==  1000.);
	BOOST_CHECK( dydt[7] == -800.);
	BOOST_CHECK( dydt[8] ==  240.);
	BOOST_CHECK( dydt[9] ==  240.);
	BOOST_CHECK( dydt[10] ==  320.);*/
}


