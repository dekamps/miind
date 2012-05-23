// Copyright (c) 2005 - 2011 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include "ParseNeatoConfig.h"

using namespace ClamLib;
using namespace std;

bool ClamLib::ParseNeatoAttributeFile
(
	const string&		filename,
	NeatoAttributes*	p_att
)
{
	const string fn = filename + ".neato";
	ifstream ifst(fn.c_str());
	if (!ifst) {
		cout << "Couldn't open  attribute file " << filename << endl;
		exit(0);
	}

	typedef boost::char_separator<char> separator_type; 
	typedef boost::tokenizer<separator_type> 	tokenizer; 
	string dummy;

	// alignment
	std::getline(ifst,dummy,'\n');
	separator_type s("<>");

	tokenizer tokena(dummy,s);
	tokenizer::iterator tok_iter = tokena.begin();
	tok_iter++;
	if (*tok_iter == string("lr") )
		p_att->_alignment = 1;
	else
		if (*tok_iter == string("rl") )
			p_att->_alignment = -1;
		else
		{
			cout << "Can't handle alignment" << endl;
			exit(0);
		}
	// node size
	std::getline(ifst,dummy,'\n');
	tokenizer tokens(dummy,s);
	tok_iter = tokens.begin();
	tok_iter++;

	p_att->_node_size = boost::lexical_cast<float>(*(tok_iter));
	
	// global offset
	std::getline(ifst,dummy,'\n');
	tokenizer tokenoffset(dummy, s);

	tok_iter = tokenoffset.begin();
	tok_iter++;
	p_att->_global_offset = boost::lexical_cast<Point>(*tok_iter);

	// perc offset
	std::getline(ifst,dummy);
	tokenizer tokenperc(dummy,s);
	tok_iter = tokenperc.begin();
	tok_iter++;
	p_att->_perc_offset = boost::lexical_cast<Point>(*tok_iter);

	// feature translation
	std::getline(ifst,dummy,'\n');
	tokenizer tokenfeatureoffset(dummy,s);

	tok_iter = tokenfeatureoffset.begin();
	tok_iter++;
	p_att->_feature_translation_vector = boost::lexical_cast<Point>(*tok_iter);

	// node translation
	std::getline(ifst,dummy,'\n');
	tokenizer tokennodeoffset(dummy,s);

	tok_iter = tokennodeoffset.begin();
	tok_iter++;
	p_att->_node_translation_vector = boost::lexical_cast<Point>(*tok_iter);

	// layer scale factor

	// node translation
	std::getline(ifst,dummy,'\n');
	tokenizer tokenlsc(dummy,s);

	tok_iter = tokenlsc.begin();
	tok_iter++;
	p_att->_layer_scale_factor = boost::lexical_cast<Point>(*tok_iter);


	return true;
}
