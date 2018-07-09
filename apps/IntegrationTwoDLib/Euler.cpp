// Copyright (c) 2005 - 2018 Marc de Kamps
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
#include "Euler.hpp"

void ClearDerivative(std::vector<double>& dydt)
{
  MPILib::Number n_dydt = dydt.size();
#pragma omp parallel for
  for (MPILib::Index ideriv = 0; ideriv < n_dydt; ideriv++)
    dydt[ideriv] = 0.;
}


void AddDerivative
(
 std::vector<double>& mass,
 const std::vector<double>& dydt,
 double h
 )
{
  MPILib::Number n_mass = mass.size();
#pragma omp parallel for
  for(MPILib::Index i = 0; i < n_mass; i++)
    mass[i] += h*dydt[i];
}


void CalculateDerivative
(
 TwoDLib::Ode2DSystemGroup&              sys,
 vector<double>&                         dydt,
 const std::vector<TwoDLib::CSRMatrix>&  vecmat,
 const std::vector<MPILib::Rate>&        vecrates
 )
{

#pragma omp parallel for

  for(MPILib::Index imat = 0; imat < vecmat.size(); imat++){  
    unsigned int nr_rows = vecmat[imat].Ia().size() - 1;
    for (MPILib::Index i = 0; i < nr_rows; i++){
      MPILib::Index mesh_index = vecmat[imat].MeshIndex();
      MPILib::Index i_r = sys.Map(i+sys.Offsets()[mesh_index]);
      for( MPILib::Index j = vecmat[imat].Ia()[i]; j < vecmat[imat].Ia()[i+1]; j++){
	int j_m = sys.Map(vecmat[imat].Ja()[j]+sys.Offsets()[mesh_index]);
	dydt[i_r] += vecrates[imat]*vecmat[imat].Val()[j]*sys.Mass()[j_m];
      }
      dydt[i_r] -= vecrates[imat]*sys.Mass()[i_r];
    }
  }
}
