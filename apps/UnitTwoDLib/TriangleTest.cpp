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

#include <boost/test/unit_test.hpp>
#include <TwoDLib.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace TwoDLib;

BOOST_AUTO_TEST_CASE(TriangleCreationTest)
{
	vector<double> vec_v(3);
	vector<double> vec_w(3);

	vec_v[0] = 0.;
	vec_v[1] = 1.;
	vec_v[2] = 0.;

	vec_w[0] = 0.;
	vec_w[1] = 0.;
	vec_w[2] = 1.;


	Triangle tri(vec_v, vec_w);

	BOOST_REQUIRE( tri.SignedArea()  ==  0.5);
	BOOST_REQUIRE( tri.IsClockwise() == -1);
}

BOOST_AUTO_TEST_CASE(TriangleInsideTest)
{
	vector<double> vec_v(3);
	vector<double> vec_w(3);

	vec_v[0] = 0.;
	vec_v[1] = 1.;
	vec_v[2] = 0.;

	vec_w[0] = 0.;
	vec_w[1] = 0.;
	vec_w[2] = 1.;

	Triangle tri(vec_v, vec_w);

	Point p1(0.1,0.1);
	BOOST_REQUIRE( tri.IsInside(p1) == 1);
	Point p2(1.0,1.0);
	BOOST_REQUIRE( tri.IsInside(p2) == 0);
}

