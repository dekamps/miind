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

#include <fstream>
#include <iomanip>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include "pugixml.hpp"
#include "Redistribution.hpp"
#include "TwoDLibException.hpp"

std::vector<TwoDLib::Redistribution> TwoDLib::ReMapping(std::istream& ifst)
{
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load(ifst);
	if (!result){
		std::cout << "Error description: " << result.description() << "\n";
		throw TwoDLib::TwoDLibException("XML parse for Remapping failed.");
	}
	if (std::string(doc.first_child().name()) != "Mapping" )
		throw TwoDLib::TwoDLibException("Mapping tag expected");

	std::istringstream  ist(doc.first_child().first_child().value());

	std::vector <TwoDLib::Redistribution> vec_reset;
	std::string from, to, fraction;
    boost::char_separator<char> sep(",");

    std::vector<unsigned int> vec_c(2,0);
    Redistribution red;
    // bug fix (MdK): 22/09/2016. Was ifst, needs to be ist, otherwise the test for goodness is meaningless
	while(ist){

		double alpha;
		ist >> from >> to >> fraction;

		if (! ist.good() ) break;

		boost::tokenizer<boost::char_separator<char>> tokensfr(from, sep);
		std::transform(tokensfr.begin(),tokensfr.end(),vec_c.begin(),boost::lexical_cast<unsigned int,std::string>);
		Coordinates cfrom(vec_c[0],vec_c[1]);
		boost::tokenizer<boost::char_separator<char>> tokensto(to, sep);
		std::transform(tokensto.begin(),tokensto.end(),vec_c.begin(),boost::lexical_cast<unsigned int,std::string>);
		Coordinates cto(vec_c[0],vec_c[1]);

		alpha = boost::lexical_cast<double>(fraction);
		red._from  = cfrom;
		red._to    = cto;
		red._alpha = alpha;

		vec_reset.push_back(red);
	}

	return vec_reset;
}


void TwoDLib::ToStream(const std::vector<Redistribution>& vec,std::ostream& s, const std::string& type){
	s << std::fixed;
	s << std::setprecision(12);
	if (type.size() > 0)
		s << "<Mapping type = \"" + type + "\">\n";
	else
		s << "<Mapping>\n";
	for(const Redistribution& r: vec)
		s << r._from[0] << "," << r._from[1] << "\t" << r._to[0] << "," << r._to[1] << "\t" << r._alpha << "\n";
	s << "</Mapping>\n";
}

