// Copyright (c) 2005 - 2008 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include "SigmoidParameter.h"

using namespace std;
using namespace NetLib;

// Default squashing function is 2/(1 + exp(-x)) - 1
// i.e. activity is in [-1,1], noise parameter beta = 1

const double F_MIN_ACT = -1;
const double F_MAX_ACT =  1;
const double F_NOISE   =  1;


SigmoidParameter::SigmoidParameter():
_f_min_act(F_MIN_ACT),
_f_max_act(F_MAX_ACT),
_f_noise(F_NOISE)
{
	_vector_of_parameters.push_back(_f_min_act);
	_vector_of_parameters.push_back(_f_max_act);
	_vector_of_parameters.push_back(_f_noise);
}

SigmoidParameter::~SigmoidParameter()
{
}

string SigmoidParameter::Tag() const
{
	//TODO: add to local definitions
	return string("<SigmoidParameter>");
}

bool SigmoidParameter::FromStream(istream& s)
{
	string tag;
	s >> tag;
	s >> _f_min_act >> _f_max_act >> _f_noise;

	s >> tag;

	return true;
	//TODO: add exception handling
}

bool SigmoidParameter::ToStream(ostream& s) const
{
	s << Tag()      << "\n";
	s << _f_min_act << "\t" << _f_max_act << "\t" << _f_noise << "\n";
	s << ToEndTag(Tag()) << "\n";

	return true;
}
/*

ostream& Connectionism::operator<<(ostream& s, const SigmoidParameter& parameter_squash)
{
	s << parameter_squash.DMinimumActivity() << "\t" 
	  << parameter_squash.DMaximumActivity() << "\t" 
	  << parameter_squash.DNoise()   << endl;

	return s;
}

istream& Connectionism::operator>>(istream& s, Connectionism::SigmoidParameter& parameter_squash)
{
	s >> parameter_squash.DMinimumActivity() >> 
		 parameter_squash.DMaximumActivity() >> 
		 parameter_squash.DNoise();

	return s;
}
*/
