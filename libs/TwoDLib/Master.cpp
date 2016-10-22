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
#include <algorithm>
#include <numeric>
#include <MPILib/include/TypeDefinitions.hpp>
#include "Master.hpp"

 using namespace TwoDLib;

 Master::Master
 (
	Ode2DSystem& sys,
	const vector<TransitionMatrix>& vec_mat,
	const MasterParameter& par
):
_sys(sys),
_vec_mat(vec_mat),
_par(par),
_dydt(sys._vec_mass.size(),0.),
_rate(0.0),
_derivative(_dydt,_sys,_rate),
_add(1./static_cast<double>(par._N_steps)),
_init(_rate)
 {
 }

 void Master::Apply(double t_step, const vector<double>& rates)
 {
	 int imat = 0;
	 for (double rate: rates){
		 _rate = t_step*rate;
		 for (MPILib::Index j = 0; j < _par._N_steps; j++){
			 std::transform(_sys._vec_mass.begin(),_sys._vec_mass.end(),_dydt.begin(),_init);
			 std::for_each(_vec_mat[imat]._vec_line.begin(),_vec_mat[imat]._vec_line.end(),_derivative);
			 std::transform(_sys._vec_mass.begin(), _sys._vec_mass.end(),_dydt.begin(),_sys._vec_mass.begin(),_add);
		 }
		 imat++;
	 }
 }
