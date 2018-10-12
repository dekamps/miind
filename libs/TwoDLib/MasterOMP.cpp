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
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <numeric>

#include "MasterOMP.hpp"

 using namespace TwoDLib;

 MasterOMP::MasterOMP
 (
	Ode2DSystem& sys,
	const vector<TransitionMatrix>& vec_mat,
	const MasterParameter& par
):
_sys(sys),
_vec_mat(vec_mat),
_vec_csr(InitializeCSR(vec_mat,sys)),
_par(par),
_dydt(sys._vec_mass.size(),0.),
_rate(0.0),
_derivative(_dydt,_sys,_rate),
_add(1./static_cast<double>(par._N_steps)),
_init(_rate)
 {
 }

 void MasterOMP::Apply(double t_step, const vector<double>& rates, const vector<MPILib::Index>& vec_map)
 {
   vector<double> mask = vector<double>(100);
   mask[50] = 1.0;

	 // the time step t_step is split into single solution steps _h, equal to 1/N_steps
	 for (unsigned int j = 0; j < floor(rates[0] * 0.001); j++){
     _vec_csr[vec_map[0]].MVCellMask
     (
        mask
     );
   }

   for (unsigned int j = 0; j < floor(rates[1] * 0.001); j++){
     _vec_csr[vec_map[0]].MVCellMaskInhib
     (
        mask
     );
   }

   vector<double> dydt = vector<double>(_sys._vec_mass.size());

   unsigned int stat = _sys.Map(0,0);
   unsigned int des = _sys.Map(72,70);
   _sys._vec_mass[des] += _sys._vec_mass[stat];
   _sys._vec_mass[stat] = 0.0;

   for (MPILib::Index j = 0; j < _sys._map[1].size(); j++) {
#pragma omp parallel for
     for (MPILib::Index i = 1; i < _sys._map.size(); i++) {
       MPILib::Index idx = _sys.Map(i,j);

       for (MPILib::Index k = 0; k < mask.size(); k++){
         int des_i = i + (k-floor(mask.size()/2));

         if ( des_i < 1 )
          des_i = 1;
         if ( des_i > _sys._map.size()-1 )
          des_i = _sys._map.size()-1;

         unsigned int idx_dest = _sys.Map(des_i, j);
         dydt[idx_dest] += _sys._vec_mass[idx] * mask[k];
       }
     }
   }

   _sys._vec_mass = vector<double>(dydt);

 }

 std::vector<CSRMatrix> MasterOMP::InitializeCSR(const std::vector<TransitionMatrix>& vec_mat, const Ode2DSystem& sys)
 {
	 std::vector<CSRMatrix> vec_ret;

	 for (const auto& mat: vec_mat)
		 vec_ret.push_back(CSRMatrix(mat,sys));

	 return vec_ret;
 }
