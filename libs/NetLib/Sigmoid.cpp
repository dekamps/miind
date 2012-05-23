
// Copyright (c) 2005 - 2007 Marc de Kamps
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

#include "Sigmoid.h"
#include "BasicDefinitions.h"
#include "LocalDefinitions.h"
#include "../NetLib/NetLib.h"

using namespace std;
using namespace NetLib;

Sigmoid::Sigmoid(){
}

Sigmoid::~Sigmoid(){
}

Sigmoid::Sigmoid(const SigmoidParameter& parameter_squash):
_parameter_squash(parameter_squash){
}

Sigmoid::Sigmoid(istream& s){
	FromStream(s);
}

bool Sigmoid::FromStream(istream& s){

	string str_sigmoid;
	s >> str_sigmoid;

	if (str_sigmoid != Tag())
		throw NetworkParsingException(STR_SIGMOID_HEADER_EXPECTED);

	s >> _parameter_squash;
 
	s >> str_sigmoid;
	if (str_sigmoid != ToEndTag(Tag()) )
		throw NetworkParsingException(STR_SIGMOID_FOOTER_EXPECTED);

	return true;
}

bool Sigmoid::ToStream(ostream& s) const {

  s << Tag() << endl;
  s << _parameter_squash;
  s << ToEndTag(Tag()) <<"\n";

  return true;
}

string Sigmoid::Tag() const {
  return STR_SIGMOID_HEADER;
}


double Sigmoid::operator()( double f ) const {
	return ( (_parameter_squash.DMaximumActivity() - _parameter_squash.DMinimumActivity())/
		     ( 1 + exp( -_parameter_squash.DNoise()*f ) ) + _parameter_squash.DMinimumActivity() ); 
}

double Sigmoid::Inverse( double f ) const {
	return (-1/_parameter_squash.DNoise())*log( (_parameter_squash.DMaximumActivity() - _parameter_squash.DMinimumActivity()) /( f - _parameter_squash.DMinimumActivity() ) - 1 );
}

double Sigmoid::Derivative( double f ) const {
	// threshold activations are artificially extreme:
	if ( fabs(f) > F_DERIVATIVE_CUTOFF )
		return 0;

	f = Inverse( f );

	double expgamma = exp( -_parameter_squash.DNoise()*f );

	return (_parameter_squash.DNoise()*( _parameter_squash.DMaximumActivity() - _parameter_squash.DMinimumActivity() )*
		expgamma/
		( ( 1 + expgamma )*( 1 + expgamma ) ) );
}

double Sigmoid::MinimumActivity() const {
	return _parameter_squash.DMinimumActivity();
}

double Sigmoid::MaximumActivity() const {
	return _parameter_squash.DMaximumActivity();
}

SigmoidParameter& Sigmoid::GetSquashingParameter() {
	return _parameter_squash;
}

AbstractSquashingFunction* Sigmoid::Clone() const {
	return new Sigmoid(*this);
}

ostream& NetLib::operator<<(ostream& s, const Sigmoid& Sig) {

	Sig.ToStream(s);
	return s;
}

istream& NetLib::operator>>(istream& s, Sigmoid& Sig) {

	Sig.FromStream(s);
	return s;
}



