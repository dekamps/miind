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

#include "MasterGrid.hpp"

 using namespace TwoDLib;

 MasterGrid::MasterGrid
 (
	Ode2DSystem& sys,
  double cell_width,
  unsigned int mask_length
):
_sys(sys),
_dydt(sys._vec_mass.size(),0.),
_mask(mask_length,0.),
_mask_swap(mask_length,0.),
_cell_width(cell_width)
 {
 }

 void MasterGrid::MVCellMask
 (
 	vector<double>&       dydt,
 	vector<double>&       dydt_new,
  double stays,
  double goes
 ) const
 {
 	for (MPILib::Index i = 0; i < dydt.size()-1; i++){
 		double mass = dydt[i];
 		dydt_new[i] += mass*stays;
 		dydt_new[i+1] += mass*goes;
 		dydt_new[i] -= mass;
 	}

 }

 void MasterGrid::MVCellMaskInhib
 (
 	vector<double>&       dydt,
 	vector<double>&       dydt_new,
  double stays,
  double goes
 ) const
 {

 	for (MPILib::Index i = 1; i < dydt.size(); i++){
 		double mass = dydt[i];
 		dydt_new[i] += mass*stays;
 		dydt_new[i-1] += mass*goes;
 		dydt_new[i] -= mass;
 	}

 }

 void MasterGrid::Apply(double t_step, const vector<double>& rates, vector<double>& efficacy_map)
 {
#pragma omp parallel for
	 for (MPILib::Index i = 0; i < _dydt.size(); i++)
		_dydt[i] = 0.0;

#pragma omp parallel for
	 for (MPILib::Index i = 0; i < _mask_swap.size(); i++) {
     _mask[i] = 0.0;
     _mask_swap[i] = 0.0;
   }

   _mask[(unsigned int)floor(_mask.size()/2)] = 1.0;
   _mask_swap[(unsigned int)floor(_mask.size()/2)] = 1.0;

   for (MPILib::Index rate = 0; rate < rates.size(); rate++)
   {
     double goes = fabs(efficacy_map[rate]/_cell_width);
     double stays = 1.0 - goes;

     for(MPILib::Index j =0; j < (floor(rates[rate] * t_step)); j++){
       if ( efficacy_map[rate] > 0 )
         MVCellMask(_mask, _mask_swap, stays, goes);
       else
         MVCellMaskInhib(_mask, _mask_swap, stays, goes);
#pragma omp parallel for
       for (MPILib::Index i = 0; i < _mask_swap.size(); i++){
         _mask[i] = _mask_swap[i];
       }
     }
   }

#pragma omp parallel for
   for (MPILib::Index j = 0; j < _sys._map[1].size(); j++) {
     for (MPILib::Index i = 0; i < _sys._map.size(); i++) {
       MPILib::Index idx = _sys.Map(i,j);

       for (MPILib::Index k = 0; k < _mask.size(); k++){
         int des_i = i + (k-floor(_mask.size()/2));

         if ( des_i < 1 )
          des_i = 1;
         if ( des_i > _sys._map.size()-1 )
          des_i = _sys._map.size()-1;

         unsigned int idx_dest = _sys.Map(des_i, j);
         _dydt[idx_dest] += _sys._vec_mass[idx] * _mask[k];
       }
     }
   }

#pragma omp parallel for
	 for (MPILib::Index imass = 0; imass < _sys._vec_mass.size(); imass++)
		 _sys._vec_mass[imass] = _dydt[imass];

 }
