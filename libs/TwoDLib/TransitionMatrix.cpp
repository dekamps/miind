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
#include <iostream>
#include <numeric>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include "MPILib/include/BasicDefinitions.hpp"
#include "MPILib/include/utilities/Log.hpp"
#include "TransitionMatrix.hpp"
#include "TwoDLibException.hpp"

using namespace TwoDLib;

TransitionMatrix::TransitionMatrix():
_vec_line(0),
_tr_v(0),
_tr_w(0)
{
}

TransitionMatrix::TransitionMatrix(const std::string& fn)
{
	std::ifstream ifst(fn);
	if (!ifst){
		std::cerr << "Can't open matrix file" << std::endl;
		LOG(MPILib::utilities::logERROR) << "Can't open matrix file";
		throw TwoDLib::TwoDLibException("Couldn't open matrix file");
	}

	boost::char_separator<char> sepsc(";");
	boost::char_separator<char> sepc(",");
	boost::char_separator<char> sepco(":");

	ifst >> _tr_v >> _tr_w;

	string dummy;
	while (ifst){
		ifst >> dummy;

		if( ifst.eof() ) break; // this is necessary or the last line will appear twice; that messes up the master equation calculation
	    boost::tokenizer< boost::char_separator<char> > tokens(dummy, sepsc);
	    std::vector<std::string> v(tokens.begin(), tokens.end());
	    if (v.size() < 2)
		throw TwoDLib::TwoDLibException("Error in line parsing from matrix files. Not enough tokens. Offending line: " + dummy);
		//to
	    boost::tokenizer< boost::char_separator<char> > tokento(v[1], sepc);
	    std::vector<std::string> vto(tokento.begin(),tokento.end());
	    if (vto.size() < 2)
		throw TwoDLib::TwoDLibException("Error in line parsing from matrix files. Not enough tokens. Offending line: " + v[1]);
	    TransferLine line;
	    line._from = Coordinates(boost::lexical_cast<unsigned int>(vto[0]),boost::lexical_cast<unsigned int>(vto[1]));
 	    for(auto it = v.begin() + 2; it != v.end(); it++){
 		    boost::tokenizer< boost::char_separator<char> > tokenco(*it, sepco);
 		    std::vector<std::string> vsp(tokenco.begin(), tokenco.end());
 		    Redistribution red;
 		    red._fraction = boost::lexical_cast<double>(vsp[1]);

 		    boost::tokenizer< boost::char_separator<char> > tokenfrom(vsp[0],sepc);
 		    std::vector<std::string> vfr(tokenfrom.begin(),tokenfrom.end());
 		    red._to = Coordinates(boost::lexical_cast<unsigned int>(vfr[0]),boost::lexical_cast<unsigned int>(vfr[1]));
 		    line._vec_to_line.push_back(red);
	    }
 	    _vec_line.push_back(line);
	}
}

bool TransitionMatrix::SelfTest(double precision) const
{

	for (const TransferLine& line: _vec_line){

		int n = line._vec_to_line.size();
		double s = 0.0;
		for (int i = 0; i < n; i++)
			s += line._vec_to_line[i]._fraction;
		if (fabs(s- 1.0) > precision) return false;
	}

	return true;
}
