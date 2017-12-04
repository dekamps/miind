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

using namespace std;
using namespace TwoDLib;
/*
BOOST_AUTO_TEST_CASE(CellCreationTest)
{
	vector<double> vec_v(4);
	vector<double> vec_w(4);

	vec_v[0] = 0.;
	vec_v[1] = 1.;
	vec_v[2] = 1.;
	vec_v[3] = 0.;

	vec_w[0] = 0.;
	vec_w[1] = 0.;
	vec_w[2] = 1.;
	vec_w[3] = 1.;

	Quadrilateral quad(vec_v, vec_w);
	BOOST_REQUIRE( quad.SignedArea()  ==  1);
	BOOST_REQUIRE( quad.IsClockwise() == -1);

	vec_v[0] = 0.;
	vec_v[1] = 0.;
	vec_v[2] = 1.;
	vec_v[3] = 1.;

	vec_w[0] = 0.;
	vec_w[1] = 1.;
	vec_w[2] = 1.;
	vec_w[3] = 0.;

	Quadrilateral quad2(vec_v, vec_w);

	BOOST_REQUIRE( quad2.SignedArea() == -1);
	BOOST_REQUIRE( quad2.IsClockwise() ==  1);

	vec_v[0] = 0.;
	vec_v[1] = 1.;
	vec_v[2] = 0.;
	vec_v[3] = 1.;

	vec_w[0] = 0.;
	vec_w[1] = 1.;
	vec_w[2] = 1.;
	vec_w[3] = 0.;

	bool b_catch = false;

	try {
		Quadrilateral quad3(vec_v,vec_w);
    } catch (TwoDLibException&) {
        b_catch = true;
    }
    BOOST_REQUIRE(b_catch);
}

BOOST_AUTO_TEST_CASE(InsideClockWise){
	vector<double>  vec_v(4);
	vector<double>  vec_w(4);

	vec_v[0] = 0.;
	vec_v[1] = 0.;
	vec_v[2] = 1.;
	vec_v[3] = 1.;

	vec_w[0] = 0.;
	vec_w[1] = 1.;
	vec_w[2] = 1.;
	vec_w[3] = 0.;

	Quadrilateral quad(vec_v, vec_w);

	Point p(0.5,0.5);


	BOOST_REQUIRE( quad.IsInside(p) == true );

	Point p1(-0.5,0.5);
	BOOST_REQUIRE(quad.IsInside(p1) == false);
	Point p2(0.5,1.5);
	BOOST_REQUIRE(quad.IsInside(p2) == false);
	Point p3(1.5,0.5);
	BOOST_REQUIRE(quad.IsInside(p3) == false);
	Point p4(0.5,-0.5);
	BOOST_REQUIRE(quad.IsInside(p4) == false);

}

BOOST_AUTO_TEST_CASE(InsideAntiClockWise){
	vector<double>  vec_v(4);
	vector<double>  vec_w(4);

	vec_v[0] = 0.;
	vec_v[1] = 1.;
	vec_v[2] = 1.;
	vec_v[3] = 0.;

	vec_w[0] = 0.;
	vec_w[1] = 0.;
	vec_w[2] = 1.;
	vec_w[3] = 1.;

	Quadrilateral quad(vec_v, vec_w);

	Point p(0.5,0.5);


	BOOST_REQUIRE( quad.IsInside(p) == true );

	Point p1(-0.5,0.5);
	BOOST_REQUIRE(quad.IsInside(p1) == false);
	Point p2(0.5,1.5);
	BOOST_REQUIRE(quad.IsInside(p2) == false);
	Point p3(1.5,0.5);
	BOOST_REQUIRE(quad.IsInside(p3) == false);
	Point p4(0.5,-0.5);
	BOOST_REQUIRE(quad.IsInside(p4) == false);
}


BOOST_AUTO_TEST_CASE(InsideConcave){
	vector<double>  vec_v(4);
	vector<double>  vec_w(4);

	vec_v[0] = -1.;
	vec_v[1] =  0.;
	vec_v[2] =  1.;
	vec_v[3] =  0.;

	vec_w[0] = -0.5;
	vec_w[1] =  0.;
	vec_w[2] = -0.5;
	vec_w[3] =  1.;

	Quadrilateral quad(vec_v, vec_w);

	Point p(0.0,0.5);
	BOOST_REQUIRE( quad.IsInside(p) == true );

	Point p1(-1.,0.);
	BOOST_REQUIRE(quad.IsInside(p1) == false);
	Point p2( 1.,0.);
	BOOST_REQUIRE(quad.IsInside(p2) == false);
	Point p3(0.,-0.1);
	BOOST_REQUIRE(quad.IsInside(p3) == false);
	Point p4(0., 1.1);
	BOOST_REQUIRE(quad.IsInside(p4) == false);

}

BOOST_AUTO_TEST_CASE(Split){
	vector<double> v(Quadrilateral::_nr_points);
	v[0] = 0;
	v[1] = 1;
	v[2] = 0;
	v[3] = -1;

	vector<double> w(Quadrilateral::_nr_points);
	w[0] = 0;
	w[1] = -0.5;
	w[2] =  1.0;
	w[3] = -0.5;

	Quadrilateral quad(v,w);

	pair<Triangle,Triangle> ts = quad.Split();

	Triangle t1 = ts.first;

	Point p1 = t1.Points()[0];
	Point p2 = t1.Points()[1];
	Point p3 = t1.Points()[2];



	BOOST_REQUIRE( p1[0] == 0 && p1[1] == 0);
	BOOST_REQUIRE( p2[0] == 0 && p2[1] == 1);
	BOOST_REQUIRE( p3[0] == 1 && p3[1] == -0.5);


	Triangle t2 = ts.second;

	p1 = t2.Points()[0];
	p2 = t2.Points()[1];
	p3 = t2.Points()[2];


	BOOST_REQUIRE( p1[0] == 0 && p1[1] == 0);
	BOOST_REQUIRE( p2[0] == 0 && p2[1] == 1);
	BOOST_REQUIRE( p3[0] == -1 && p3[1] == -0.5);

// Now rotate the points. Same triangles should be produced

	v[1] = 0;
	v[2] = 1;
	v[3] = 0;
	v[0] = -1;

	w[1] = 0;
	w[2] = -0.5;
	w[3] =  1.0;
	w[0] = -0.5;

	Quadrilateral quad2(v,w);
	pair<Triangle,Triangle> ts2 = quad2.Split();

	Triangle t3 = ts2.first;

	p1 = t3.Points()[0];
	p2 = t3.Points()[1];
	p3 = t3.Points()[2];


	BOOST_REQUIRE( p1[0] == 0 && p1[1] == 0);
	BOOST_REQUIRE( p2[0] == 0 && p2[1] == 1);
	BOOST_REQUIRE( p3[0] == -1 && p3[1] == -0.5);

	Triangle t4 = ts2.second;

	p1 = t4.Points()[0];
	p2 = t4.Points()[1];
	p3 = t4.Points()[2];

	BOOST_REQUIRE( p1[0] == 0 && p1[1] == 0);
	BOOST_REQUIRE( p2[0] == 0 && p2[1] == 1);
	BOOST_REQUIRE( p3[0] == 1 && p3[1] == -0.5);

}

BOOST_AUTO_TEST_CASE(CentroidTest){
	vector<double>  vec_v(4);
	vector<double>  vec_w(4);

	vec_v[0] = 0.;
	vec_v[1] = 0.;
	vec_v[2] = 1.;
	vec_v[3] = 1.;

	vec_w[0] = 0.;
	vec_w[1] = 1.;
	vec_w[2] = 1.;
	vec_w[3] = 0.;

	Quadrilateral quad(vec_v, vec_w);

	Point centre = quad.Centroid();
	BOOST_REQUIRE(centre[0] == 0.5);
	BOOST_REQUIRE(centre[1] == 0.5);
}
*/
