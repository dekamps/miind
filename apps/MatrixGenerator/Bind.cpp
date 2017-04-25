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
#include <string>
#include <vector>
#include "Bind.hpp"
#include "TwoDLibException.hpp"

namespace {

	void Glue(
			const std::string& base_name,
			const TwoDLib::Mesh& mesh,
			const TwoDLib::Stat& stat,
			const std::vector<TwoDLib::Redistribution>& vec_rev,
			double V_res,
			double theta)
	{
		std::ofstream ofst(base_name + ".model");
		ofst << "<Model>\n";
		mesh.ToXML(ofst);
		stat.ToXML(ofst);
		TwoDLib::ToStream(vec_rev,ofst,"Reversal");
		ofst << "<threshold>" << theta << "</threshold>\n";
		ofst << "<V_reset>"   << V_res << "</V_reset>\n";
		ofst << "</Model>\n";
	}
}

void TwoDLib::Bind(int argc,char** argv){
	std::cout << "Creating model file" << std::endl;

	std::string mesh_name(argv[1]);
	std::vector<string> elem;
	split(mesh_name,'.',elem);
	if (elem.size() < 2 || elem[1] != string("mesh"))
		throw TwoDLib::TwoDLibException("Mesh extension not .mesh");
	std::string base_name = elem[0];
	elem.clear();
	std::string stat_name(argv[2]);
	split(stat_name,'.',elem);
	if (elem.size() < 2 || elem[1] != string("stat"))
		throw TwoDLib::TwoDLibException("Stat extension not .stat");
	string rev_name(argv[3]);
	elem.clear();
	split(rev_name,'.',elem);
	if (elem.size() < 2 || elem[1] != string("rev"))
		throw TwoDLib::TwoDLibException("rev file extension not .rev");

	std::cout << "Reading mesh" << std::endl;
	TwoDLib::Mesh mesh(base_name + ".mesh");
	std::cout << "There are " << mesh.NrQuadrilateralStrips() << " strips." << std::endl;
	std::cout << "Reading stat file" << std::endl;
	TwoDLib::Stat stat(base_name + ".stat");

	std::cout << "Inserting stationary bins" << std::endl;
	// now loop over stationary lines and insert them into the mesh
	std::vector<TwoDLib::Quadrilateral> vec_quad = stat.Extract();
	for (const Quadrilateral& quad: vec_quad)
		mesh.InsertStationary(quad);

	std::cout << "Reading rev file" << std::endl;
	std::ifstream ifrev(base_name + ".rev");
	if (! ifrev)
		throw TwoDLibException("Couldn't open .rev file.");
	std::vector<TwoDLib::Redistribution> vec_rev =  TwoDLib::ReMapping(ifrev);

	std::cout << "Reset potential: ";
	// reset potential
	std::istringstream ist_vres(argv[4]);
	double V_res;
	ist_vres >> V_res;
	std::cout << V_res << std::endl;

	std::cout << "Reversal potential: ";
	// threshold potential
	std::istringstream ist_theta(argv[5]);
	double theta;
	ist_theta >> theta;
	std::cout << theta << std::endl;

	Glue(base_name,mesh,stat,vec_rev, V_res, theta);
	std::cout << "Model file created" << std::endl;
}


void InterpretArguments(int argc, char** argv){
	if (argc != 6){
		std::cout << "Usage: ./Bind <basename>.mesh <basename>.stat <basename>.rev V_res theta." << std::endl;
		exit(0);
	}
}

int main(int argc, char** argv){

	try {

		InterpretArguments(argc,argv);
		TwoDLib::Bind(argc,argv);
	}
	catch(const TwoDLib::TwoDLibException& excep){
		std::cout << excep.what() << std::endl;
	}

	return 0;
}
