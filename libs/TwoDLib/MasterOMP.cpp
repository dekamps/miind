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
#include <random>
#include <fstream>

#include "MasterOMP.hpp"

 using namespace TwoDLib;

 MasterOMP::MasterOMP
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
_rate(0.0),
_derivative(_dydt,_sys,_rate),
_add(1./static_cast<double>(par._N_steps)),
_init(_rate)
 {
 }

 void MasterOMP::Apply
 (
	double t_step,
	const std::vector< std::vector<double > >& vec_vec_rates,
	const vector<MPILib::Index>&               vec_vec_map
)
 {
	 MPILib::Number n_mesh = vec_vec_rates.size();
	 // the time step t_step is split into single solution steps _h, equal to 1/N_steps
	 for (unsigned int j = 0; j < _par._N_steps; j++){

#pragma omp parallel for
		 for(int id = 0; id < _dydt.size(); id++)
			 _dydt[id] = 0.;
		 for(MPILib::Index mesh_index = 0; mesh_index < n_mesh; mesh_index++){
			 for (unsigned int irate = 0; irate < vec_vec_rates[mesh_index].size(); irate++){
				 // do NOT map the rate
				 _rate = vec_vec_rates[mesh_index][irate];
				 // it is only the matrices that need to be mapped
				 _vec_vec_csr[mesh_index][vec_vec_map[irate]].MVMapped
				 (
					_dydt,
					_sys._vec_mass,
					_rate
				 );
			 }
		 }

#pragma omp parallel for
		 for (int imass = 0; imass < _sys._vec_mass.size(); imass++)
			 _sys._vec_mass[imass] += _add._h*t_step*_dydt[imass]; // the mult
	 }
 }

 void MasterOMP::ApplyFinitePoisson
 (
	 double t_step,
	 const std::vector< std::vector<double > >& vec_vec_rates,
	 const vector<MPILib::Index>& vec_vec_map
 )
 {
	 static std::random_device rd;
	 static std::mt19937 gen(rd());
	 MPILib::Number n_mesh = vec_vec_rates.size();

#pragma omp parallel for
	 for (int id = 0; id < _sys._vec_objects_to_index.size(); id++) {
		 if (_sys._vec_objects_refract_times[id] >= 0.0)
			 continue;

		 for (MPILib::Index mesh_index = 0; mesh_index < n_mesh; mesh_index++) {
			 for (unsigned int irate = 0; irate < vec_vec_rates[mesh_index].size(); irate++) {
				 if (vec_vec_rates[mesh_index][irate] == 0)
					 continue;

				 std::poisson_distribution<int> pd(vec_vec_rates[mesh_index][irate] * t_step);
				 int spikes = pd(gen);
				 _sys._vec_objects_to_index[id] =
					 _vec_vec_csr[mesh_index][vec_vec_map[irate]].MVObject(_sys._vec_objects_to_index[id], spikes);
			 }
		 }
	 }

	 _sys.updateVecCellsToObjects();
 }

std::vector<std::vector<CSRMatrix> > MasterOMP::InitializeCSR(const std::vector< std::vector<TransitionMatrix> >& vec_vec_mat, const Ode2DSystemGroup& sys)
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
