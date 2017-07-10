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
#include <iostream>
#include <fstream>

using namespace std;
using namespace TwoDLib;

BOOST_AUTO_TEST_CASE(TriangleGeneratorTest)
{

	Point p1(0.,0.);
	Point p2(1.,0.);
	Point p3(0.,1.);
	Triangle t(p1,p2,p3);

	Uniform uni(0);
	TriangleGenerator gen(t,uni);
	vector<Point> vec(10);

	gen.Generate(&vec);

}

BOOST_AUTO_TEST_CASE(QuadGeneratorTest)
{
	Point p1(0.,0.);
	Point p2(1.,0.);
	Point p3(1.,1.);
	Point p4(0.,1.);

	Quadrilateral q(p1,p2,p3,p4);

	Uniform uni(0);
	QuadGenerator gen(q,uni);
	vector<Point> vec(1000);

	gen.Generate(&vec);

	ofstream ofst("square.dat");
	for (auto it = vec.begin(); it != vec.end(); it++)
		ofst << (*it)[0] << " " << (*it)[1] << '\n';
	ofst.close();

	Point p5(0.,0.);
	Point p6(1,-0.5);
	Point p7(0.,1);
	Point p8(-0.5,-0.5);

	Quadrilateral kite(p5,p6,p7,p8);
	QuadGenerator genkite(kite,uni);
	genkite.Generate(&vec);

	ofstream ofst2("kite.dat");
	for (auto it = vec.begin(); it != vec.end(); it++)
		ofst2 << (*it)[0] << " " << (*it)[1] << '\n';
	ofst2.close();

}
