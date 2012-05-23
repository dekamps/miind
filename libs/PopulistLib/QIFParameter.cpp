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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "QIFParameter.h"
#include "LocalDefinitions.h"
#include "PopulistException.h"

using namespace PopulistLib;

string QIFParameter::Tag() const
{
	return "<QIFParameter>";
}

bool QIFParameter::ToStream(ostream& s) const
{
	string dummy;
	s << this->Tag()						<< "\n";
	s << this->WrapTag(_V_min, "V_min")		<< "\n";
	s << this->WrapTag(_V_peak, "V_peak")	<< "\n";
	s << this->WrapTag(_V_reset, "V_reset")	<< "\n";
	s << this->WrapTag(_I,"I")				<< "\n";
	s << this->ToEndTag(this->Tag())		<< "\n";

	return true;
}

bool QIFParameter::FromStream(istream& s)
{
	string dummy;

	s >> dummy;
	if (dummy != Tag() )
		throw PopulistException(STR_QIFP_UNEXPECTED);

	s >> dummy;
	_V_min =  boost::lexical_cast<Potential>(this->UnWrapTag(dummy));
	s >> dummy;
	_V_peak =  boost::lexical_cast<Potential>(this->UnWrapTag(dummy));
	s >> dummy;
	_V_reset =  boost::lexical_cast<Potential>(this->UnWrapTag(dummy));
	s>> dummy;
	_I =  boost::lexical_cast<Potential>(this->UnWrapTag(dummy));

	s >> dummy;
	if (dummy != this->ToEndTag(this->Tag()))
		throw PopulistException(STR_QIFP_UNEXPECTED);



	return true;
}


bool PopulistLib::operator==(const QIFParameter& q1, const QIFParameter& q2)
{
	return (q1._V_min == q2._V_min) && (q1._V_peak == q2._V_peak) && (q1._V_reset == q2._V_reset) && (q1._I == q2._I);
}
