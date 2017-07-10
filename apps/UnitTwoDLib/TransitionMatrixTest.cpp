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

using namespace std;
using namespace TwoDLib;

BOOST_AUTO_TEST_CASE(TransitionMatrixConstructionTest)
{
	TransitionMatrix mat("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mat");
}


BOOST_AUTO_TEST_CASE(SimpleMeshTest){
	try {
		Mesh m("simple.mesh");
		vector<Redistribution> vec_dummy;
		Ode2DSystem sys(m,vec_dummy,vec_dummy);

		TransitionMatrix mat("simple.mat");
		CSRMatrix csrmat(mat,sys);

		vector<double> out{ 0., 0., 0., 0., };
		vector<double> v{ 1.0, 0., 0., 0. };

		csrmat.MV(out,v);

		BOOST_REQUIRE(out[0] == 0.5);
		BOOST_REQUIRE(out[1] == 0.0);
		BOOST_REQUIRE(out[2] == 0.5);
		BOOST_REQUIRE(out[3] == 0.0);

	}
	catch(const TwoDLibException& excep){
		std::cout << excep.what() << std::endl;
	}
}

