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

BOOST_AUTO_TEST_CASE(TestSingleTransitionGeneration)
{
	Uniform uni(789);
	Mesh mesh("aexpoverview.mesh");

	MeshTree tree(mesh);
	TransitionMatrixGenerator gen(tree,uni);

	gen.GenerateTransition(10, 4, 0.,30.);

	vector<Point> lost = gen.LostPoints();

	BOOST_REQUIRE( lost.size() == 0);

	vector<TransitionMatrixGenerator::Hit> hit_list = gen.HitList();
	BOOST_REQUIRE( hit_list.size() == 2    );
	BOOST_REQUIRE( hit_list[0]._count == 3 );
	BOOST_REQUIRE( hit_list[1]._count == 7 );
}

BOOST_AUTO_TEST_CASE(FiducialTest){

	Mesh mesh("aexpoverview.mesh");
	Uniform uni(789);

	vector<Point> perim(4);
	perim[0][0] = -70.9;
	perim[0][1] =  0.;
	perim[1][0] = -73.4;
	perim[1][1] =  69.0;
	perim[2][0] = -72.6;
	perim[2][1] =  69.0;
	perim[3][0] = -70.1;
	perim[3][1] =  0.;

	Quadrilateral quad(perim);
	vector<Coordinates> vec_fiducial = mesh.CellsBelongTo(quad);
	FiducialElement el(mesh,quad,LEAK,vec_fiducial);
	FidElementList list(vector<FiducialElement>(1,el));

	MeshTree tree(mesh);
	TransitionMatrixGenerator gen(tree,uni,1000,list);
	gen.GenerateTransition(3, 0, 7.,0.);

	vector<TransitionMatrixGenerator::Hit> hit_list = gen.HitList();
	int sum = 0;
	for (auto it = hit_list.begin(); it != hit_list.end(); it++){
		sum += it->_count;
	}
	BOOST_REQUIRE(sum == 882);
	BOOST_REQUIRE(gen.AccountedPoints().size() == 118);
	BOOST_REQUIRE(gen.LostPoints().size() == 0);
}

BOOST_AUTO_TEST_CASE(FiducialContainTest){

	Mesh mesh("aexpoverview.mesh");
	Uniform uni(789);

	vector<Point> perim(4);
	perim[0][0] = -70.9;
	perim[0][1] =  0.;
	perim[1][0] = -73.4;
	perim[1][1] =  69.0;
	perim[2][0] = -72.6;
	perim[2][1] =  69.0;
	perim[3][0] = -70.1;
	perim[3][1] =  0.;

	Quadrilateral quad(perim);
	vector<Coordinates> vec_fiducial = mesh.CellsBelongTo(quad);
	FiducialElement el(mesh,quad,CONTAIN,vec_fiducial);
	FidElementList list(vector<FiducialElement>(1,el));

	MeshTree tree(mesh);
	TransitionMatrixGenerator gen(tree,uni,1000,list);
	gen.GenerateTransition(3, 0, 7.,0.);

	vector<TransitionMatrixGenerator::Hit> hit_list = gen.HitList();
	int sum = 0;
	for (auto it = hit_list.begin(); it != hit_list.end(); it++){
		sum += it->_count;
	}

	BOOST_REQUIRE(sum == 1000);
	BOOST_REQUIRE(gen.AccountedPoints().size() == 0);
	BOOST_REQUIRE(gen.LostPoints().size() == 0);
}


