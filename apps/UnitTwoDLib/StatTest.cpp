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

BOOST_AUTO_TEST_CASE(StatTest)
{
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file("test.stat");
	pugi::xml_node node = doc.first_child();

	BOOST_REQUIRE(std::string(node.name()) == "Stationary");

	pugi::xml_node quad = node.first_child();
	BOOST_REQUIRE(std::string(quad.name()) == "Quadrilateral");

	pugi::xml_node vline = quad.first_child();
	BOOST_REQUIRE(std::string(vline.name()) == "vline");

	pugi::xml_node values = vline.first_child();
	std::istringstream  ist(values.value());
	float x,y,z,p;
	ist >> x >> y >> z >> p;

	pugi::xml_node wline = vline.next_sibling();

	std::istringstream ist2(wline.first_child().value());
	ist2 >> x >> y >> z >> p;

}


BOOST_AUTO_TEST_CASE(StatTest2)
{
	// push too far to see error handling
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file("test.stat");
	pugi::xml_node node = doc.first_child();

	BOOST_REQUIRE(std::string(node.name()) == "Stationary");

	pugi::xml_node quad = node.first_child();
	pugi::xml_node vline = quad.first_child();
	pugi::xml_node value = vline.first_child();
	pugi::xml_node wrong = value.first_child();
	BOOST_REQUIRE( wrong.type() == pugi::node_null);

}

BOOST_AUTO_TEST_CASE(StatTest3)
{
	// iterate through all stationary points
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file("multiple.stat");
	pugi::xml_node stat = doc.first_child();

	BOOST_REQUIRE(result.status == pugi::status_ok);

	for (pugi::xml_node quad = stat.first_child(); quad; quad = quad.next_sibling()){
		BOOST_REQUIRE(std::string(quad.first_child().first_child().value()) == "0. 0. 1. 1.");
	}

}

BOOST_AUTO_TEST_CASE(StatTest4)
{
	// iterate through all staionary points
	Stat stat( "multiple.stat" );

	std::vector<Quadrilateral> vec_quad = stat.Extract();
	BOOST_REQUIRE(vec_quad.size() == 3);

}


BOOST_AUTO_TEST_CASE(FidTest)
{
	Mesh mesh("aexp0cfa5d69-a740-4474-b8b6-b39870e2f5ef.mesh");
	Fid fid( "aexp0cfa5d69-a740-4474-b8b6-b39870e2f5ef.fid" );
	std::vector<ProtoFiducial> vec_fid = fid.Extract();

	BOOST_REQUIRE(vec_fid.size() == 9);
	BOOST_REQUIRE(vec_fid[0].second == CONTAIN);
	BOOST_REQUIRE(vec_fid[4].second == LEAK);

	std::vector<FiducialElement> list = fid.Generate(mesh);

	BOOST_REQUIRE(list.size() == 9);
}
