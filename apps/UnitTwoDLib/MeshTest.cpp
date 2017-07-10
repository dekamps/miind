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

BOOST_AUTO_TEST_CASE(MeshTest)
{
	// Use aexpoverview.mesh for detailed debugging. It is small enough
	// to see every cell. Use cond.mesh for large scale testing.
/*	Mesh mesh("cond.mesh");
	unsigned int nr_strip = mesh.NrQuadrilateralStrips();
	BOOST_REQUIRE(nr_strip == 199);

	unsigned int nr_cells = mesh.NrCellsInStrip(88);
	BOOST_REQUIRE(nr_cells == 1000);*/
}


BOOST_AUTO_TEST_CASE(PoinTbelongsToTest){

	Mesh mesh("aexpoverview.mesh");

	Quadrilateral quad = mesh.Quad(10,10);
	vector<Point> vec = quad.Points();
	Point p = vec[0];
 	vector<Coordinates> vecid = mesh.PointBelongsTo(p);

	BOOST_REQUIRE(vecid.size() == 4);

	BOOST_REQUIRE( vecid[0][0]  == 10);
	BOOST_REQUIRE( vecid[0][1] == 9);

	BOOST_REQUIRE( vecid[1][0]  == 10);
	BOOST_REQUIRE( vecid[1][1] == 10);

	BOOST_REQUIRE( vecid[2][0]  == 37);
	BOOST_REQUIRE( vecid[2][1] == 9);

	BOOST_REQUIRE( vecid[3][0]   == 37);
	BOOST_REQUIRE( vecid[3][1]  == 10 );
}

BOOST_AUTO_TEST_CASE(CellsBelongToTest){
	Mesh mesh("aexpoverview.mesh");

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
	std::cout << vec_fiducial.size() << std::endl;
}

BOOST_AUTO_TEST_CASE(XMLTest){

	Mesh mesh("simple.mesh");
	ofstream ofst("simplexml.mesh");
	mesh.ToXML(ofst);
}

BOOST_AUTO_TEST_CASE(ReadXML){

	Mesh mesh("simplexml.mesh");
	ofstream ofst("simplexml2.mesh");
	mesh.ToXML(ofst);
}

BOOST_AUTO_TEST_CASE(BiggerMeshXML){
	// initially this test failed. It turned out to be important to set
	// a high precision on the output stream in Mesh::ToXML.

	Mesh mesh("aexpoverview.mesh");

	std::ofstream ofst("aexpoverviewXML.mesh");
	mesh.ToXML(ofst);
	ofst.close();

	std::ifstream ifst("aexpoverviewXML.mesh");
	try {
		Mesh mesh2(ifst);
	}
	catch(const TwoDLib::TwoDLibException& excep){
		std::cout << excep.what() << std::endl;
	}

}
