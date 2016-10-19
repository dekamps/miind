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
///*
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include "Stat.hpp"
#include "TwoDLibException.hpp"

using namespace TwoDLib;

void Stat::Load(const pugi::xml_node& stat){
	vector<double> v(Quadrilateral::_nr_points);
	vector<double> w(Quadrilateral::_nr_points);


	for (pugi::xml_node quad = stat.first_child(); quad; quad = quad.next_sibling()){

		pugi::xml_node element = quad.first_child();
		pugi::xml_node vline;


		if (element.first_child().type() != pugi::node_null){
			vline = element.first_child();
			std::istringstream ist(vline.value());
			ist >> v[0] >> v[1] >> v[2] >> v[3];
		}
		else
			throw TwoDLibException("Can't read .stat file");

		pugi::xml_node wline = element.next_sibling();
		if (wline.first_child().type() != pugi::node_null){
			std::istringstream ist(wline.first_child().value());
			ist >> w[0] >> w[1] >> w[2] >> w[3];
		}
		else
			throw TwoDLibException("Can't read .stat file");
		Quadrilateral kwad(v,w);
		_vec_quad.push_back(kwad);
	}
}

Stat::Stat(std::istream& s){
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load(s);
	pugi::xml_node stat = doc.first_child();

	if (result.status != pugi::status_ok)
		throw TwoDLibException("Can't open .stat file.");

	this->Load(stat);
}

Stat::Stat(const std::string& fn){

	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(fn.c_str());
	pugi::xml_node stat = doc.first_child();

	if (result.status != pugi::status_ok)
		throw TwoDLibException("Can't open .stat file.");
	this->Load(stat);
}

void Stat::ToXML(std::ostream& s) const {
	s << std::setprecision(10);
	s << "<Stationary>\n";
	for( const Quadrilateral quad: _vec_quad){
		s << "<Quadrilateral>";
		s << "<vline>";
		for(const Point& p: quad.Points())
			s << p[0] << " ";
		s << "</vline>";
		s << "<wline>";
		for(const Point& p: quad.Points())
			s << p[1] << " ";
		s << "</wline>";
		s << "</Quadrilateral>\n";
	}

	s << "</Stationary>\n";
}
