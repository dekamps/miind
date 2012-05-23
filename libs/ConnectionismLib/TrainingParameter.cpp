// Copyright (c) 2005 - 2009 Marc de Kamps
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
#include "TrainingParameter.h"

using namespace ConnectionismLib;


TrainingParameter::TrainingParameter():
	_f_stepsize(0),
	_f_sigma(0),
	_f_bias(0),
	_n_step(0),
	_train_threshold(false),
	_f_momentum(0),
	_l_seed(0),
	_n_init(0)
{
}

		
TrainingParameter::TrainingParameter( double f_stepsize,
				   double f_sigma,
				   double f_bias,
				   size_t n_step,
				   bool   b_train_threshold,
				   double f_threshold_default,
				   double f_momentum,
				   long   l_seed,
				   size_t n_init ):
_f_stepsize(f_stepsize),
_f_sigma   (f_sigma),
_f_bias    (f_bias ),
_n_step    (n_step),
_train_threshold(b_train_threshold),
_f_momentum(f_momentum),
_l_seed    (l_seed),
_n_init    (n_init)
{}

ostream& ConnectionismLib::operator<<(ostream& s, const TrainingParameter& par_train)
{
	s << par_train._f_momentum << "\n";
	s << par_train._f_sigma << "\n";
	s << par_train._f_stepsize << "\n";
	s << par_train._l_seed << "\n";
	s << static_cast<unsigned int>(par_train._n_init) << "\n";
	s << static_cast<unsigned int>(par_train._n_step) << "\n";
	s << par_train._train_threshold << "\n";

	return s;
}

istream& ConnectionismLib::operator>>(istream& s, TrainingParameter& par_train)
{
	s >> par_train._f_momentum;
	s >> par_train._f_sigma;
	s >> par_train._f_stepsize;
	s >> par_train._l_seed;
	s >> par_train._n_init;
	s >> par_train._n_step;
	s >> par_train._train_threshold;

	return s;
}
