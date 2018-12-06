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
#include "Fid.hpp"
#include "FiducialElement.hpp"
#include "pugixml.hpp"
#include "TwoDLibException.hpp"
#include "Quadrilateral.hpp"

using namespace TwoDLib;

Fid::Fid(const std::string& fn){

	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(fn.c_str());
	pugi::xml_node stat = doc.first_child();

	if (result.status != pugi::status_ok)
		throw TwoDLibException("Can't open .fid file.");

	vector<double> v(Quadrilateral::_nr_points);
	vector<double> w(Quadrilateral::_nr_points);


	for (pugi::xml_node quad = stat.first_child(); quad; quad = quad.next_sibling()){
		int i = 0;
		Overflow over;
	    for (auto ait : quad.attributes()){
	    	if (i > 0) throw TwoDLibException("Expected one attribute in Fiducial.");

	        if (std::string(ait.value()) == std::string("Contain") )
	        	over = CONTAIN;
	        else if (std::string(ait.value()) == std::string("Leak") )
	        	over = LEAK;
	        else
	        	throw TwoDLibException("Unknown type of FiducialElement");
	        i++;
	    }

		pugi::xml_node element = quad.first_child();
		pugi::xml_node vline;

		if (element.first_child().type() != pugi::node_null){
			vline = element.first_child();
			std::istringstream ist(vline.value());
			ist >> v[0] >> v[1] >> v[2] >> v[3];
		}
		else
			throw TwoDLibException("Can't read .fid file");

		pugi::xml_node wline = element.next_sibling();
		if (wline.first_child().type() != pugi::node_null){
			std::istringstream ist(wline.first_child().value());
			ist >> w[0] >> w[1] >> w[2] >> w[3];
		}
		else
			throw TwoDLibException("Can't read .stat file");
		Cell kwad(v,w);

		ProtoFiducial prot(kwad,over);
		_vec_prot.push_back(prot);
	}
}

std::vector<FiducialElement> Fid::Generate(const Mesh& mesh) const
{
	vector<TwoDLib::FiducialElement> vec_elements;
	for (const auto& prot: _vec_prot){
		vector<TwoDLib::Coordinates> vec_fiducial = mesh.CellsBelongTo(prot.first);
		TwoDLib::FiducialElement el(mesh,prot.first,prot.second,vec_fiducial);
		vec_elements.push_back(el);
	}

	return vec_elements;
}
