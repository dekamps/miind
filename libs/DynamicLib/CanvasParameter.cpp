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

#include <boost/lexical_cast.hpp>
#include "CanvasParameter.h"
#include "DynamicLibException.h"

using namespace DynamicLib;

CanvasParameter::CanvasParameter():
_t_min		(0),
_t_max		(0),
_f_min		(0),
_f_max		(0),
_state_min	(0),
_state_max	(0),
_dense_min	(0),
_dense_max	(0)
{
}

CanvasParameter::CanvasParameter
(
	double t_min,
	double t_max,
	double f_min,
	double f_max,
	double state_min,
	double state_max,
	double dense_min,
	double dense_max
):
_t_min(t_min),
_t_max(t_max),
_f_min(f_min),
_f_max(f_max),
_state_min(state_min),
_state_max(state_max),
_dense_min(dense_min),
_dense_max(dense_max)
{
}

CanvasParameter::~CanvasParameter()
{
}

string CanvasParameter::Tag() const
{
	return "<CanvasParameter>";
}

bool CanvasParameter::ToStream(ostream& s) const
{
	s << Tag() << "\n";

	s << this->WrapTag(_t_min,"T_min")			<< "\n";
	s << this->WrapTag(_t_max,"T_max")			<< "\n";
	s << this->WrapTag(_f_min, "F_max")			<< "\n";
	s << this->WrapTag(_f_max, "F_max")			<< "\n";
	s << this->WrapTag(_state_min,"State_min")	<< "\n";
	s << this->WrapTag(_state_max,"State_max")	<< "\n";
	s << this->WrapTag(_dense_min, "Dense_max")	<< "\n";
	s << this->WrapTag(_dense_max, "Dense_max")	<< "\n";
	s << this->ToEndTag(this->Tag()) << "\n";

	return true;
}

bool CanvasParameter::FromStream(istream& s)
{
	string dummy;
	s >> dummy;

	if (dummy != this->Tag() )
		throw DynamicLibException("Expected CanvasParameter tag");

	s >> dummy;
	_t_min = boost::lexical_cast<double>(this->UnWrapTag(dummy));
	s >> dummy;
	_t_max = boost::lexical_cast<double>(this->UnWrapTag(dummy));
	s >> dummy;
	_f_min = boost::lexical_cast<double>(this->UnWrapTag(dummy));
	s >> dummy;
	_f_max = boost::lexical_cast<double>(this->UnWrapTag(dummy));
	s >> dummy;
	_state_min = boost::lexical_cast<double>(this->UnWrapTag(dummy));
	s >> dummy;
	_state_max = boost::lexical_cast<double>(this->UnWrapTag(dummy));
	s >> dummy;
	_dense_min = boost::lexical_cast<double>(this->UnWrapTag(dummy));
	s >> dummy;
	_dense_max = boost::lexical_cast<double>(this->UnWrapTag(dummy));

	s >> dummy;
	if (dummy != this->ToEndTag(this->Tag()))
		throw DynamicLibException("CanvasParameter end tag expected");

	return true;
}