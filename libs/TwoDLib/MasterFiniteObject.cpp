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
#include <fstream>

#include "MasterFiniteObject.hpp"

 using namespace TwoDLib;

 MasterFiniteObject::MasterFiniteObject
 (
	Ode2DSystemGroup&                                  sys,
	const std::vector<std::vector<TransitionMatrix> >& vec_vec_mat,
	const MasterParameter& par
):
_sys(sys),
_vec_vec_mat(vec_vec_mat),
_vec_vec_csr(InitializeCSR(vec_vec_mat,sys)),
_par(par),
_dydt(sys._vec_mass.size(),0.),
_rate(0.0)
 {
 }

 void MasterFiniteObject::Apply
 (
	double t_step,
	const std::vector< std::vector<double > >& vec_vec_rates,
	const vector<MPILib::Index>&               vec_vec_map
)
 {
	 MPILib::Number n_mesh = vec_vec_rates.size();

#pragma omp parallel for
	 for (int id = 0; id < _sys._vec_objects_to_index.size(); id++) {
		 for (MPILib::Index mesh_index = 0; mesh_index < n_mesh; mesh_index++) {
			 for (unsigned int irate = 0; irate < vec_vec_rates[mesh_index].size(); irate++) {
				 _sys._vec_objects_to_index[id] = _vec_vec_csr[mesh_index][vec_vec_map[irate]].MVObject(_sys._vec_objects_to_index[id], 1);
			 }
		 }
	 }

	_sys.updateVecCellsToObjects();
 }

std::vector<std::vector<CSRMatrix> > MasterFiniteObject::InitializeCSR(const std::vector< std::vector<TransitionMatrix> >& vec_vec_mat, const Ode2DSystemGroup& sys)
  {
 	 std::vector<std::vector<CSRMatrix> > vec_ret;
 	 MPILib::Index mesh_index = 0;
 	 for (const auto& vec_mat: vec_vec_mat){
 		 std::vector<CSRMatrix> vec_dummy;
 		 for (const auto& mat: vec_mat)
 			 vec_dummy.push_back(CSRMatrix(mat,sys,mesh_index));
 		 vec_ret.push_back(vec_dummy);
 		 mesh_index++;
 	 }
 	 return vec_ret;
  }
